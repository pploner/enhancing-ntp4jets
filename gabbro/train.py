import glob
import hashlib
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import pyrootutils


def _load_env_file(path: str) -> None:
    """Load a simple KEY=VALUE env file, overriding existing values."""
    env_path = Path(os.path.expandvars(os.path.expanduser(path)))
    if not env_path.is_file():
        raise FileNotFoundError(f"GABBRO_ENV_FILE does not exist: {env_path}")

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = os.path.expandvars(value)


def _load_explicit_env_file() -> None:
    env_file = os.environ.get("GABBRO_ENV_FILE")
    if env_file:
        _load_env_file(env_file)


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
_load_explicit_env_file()

try:
    import comet_ml  # noqa: F401  # import before torch when Comet logging is installed
except ImportError:
    comet_ml = None
import hydra
import lightning as L
import torch
from hydra import compose
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import get_rank, get_world_size
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
# - optionally loading a user-selected env file through GABBRO_ENV_FILE
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import gabbro.models.lightning_models as gabbro_lightning_models
import gabbro.utils.git_utils as git_utils
from gabbro.utils.bigram import get_bigram
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import (
    get_gpu_properties,
    instantiate_callbacks,
    instantiate_loggers,
    get_metric_value,
    log_hyperparameters,
    remove_empty_hydra_run_dir,
    task_wrapper,
)


log = get_pylogger(__name__)


def _load_feature_expanded_weights(
    model: L.LightningModule,
    source_state: dict[str, torch.Tensor],
    source_cfg: DictConfig,
    target_cfg: DictConfig,
    expansion_cfg: DictConfig,
) -> None:
    """Warm-start a model whose ordered continuous feature set was expanded."""
    target_state = model.state_dict()
    source_features = list(source_cfg.feature_dict.keys())
    target_features = list(target_cfg.feature_dict.keys())
    source_pid = source_cfg.get("pid") or {}
    target_pid = target_cfg.get("pid") or {}
    source_pid_count = int(source_pid.get("num_classes", 0)) if source_pid.get("enabled") else 0
    target_pid_count = int(target_pid.get("num_classes", 0)) if target_pid.get("enabled") else 0
    if source_pid_count != target_pid_count:
        raise ValueError(
            "Feature-expanded warm start requires unchanged PID conditioning; got "
            f"{source_pid_count} and {target_pid_count} classes"
        )
    if not set(source_features).issubset(target_features):
        raise ValueError(
            "Feature-expanded warm start only supports additive features; source="
            f"{source_features}, target={target_features}"
        )

    input_key = "model.input_projection.weight"
    output_weight_key = "model.output_projection.weight"
    output_bias_key = "model.output_projection.bias"
    allowed_shape_changes = {input_key, output_weight_key, output_bias_key}
    unexpected = sorted(set(source_state) - set(target_state))
    if unexpected:
        raise ValueError(f"Warm-start checkpoint contains unexpected tensors: {unexpected}")

    expanded_state = {key: value.clone() for key, value in target_state.items()}
    mismatched = []
    for key, source_value in source_state.items():
        if source_value.shape == target_state[key].shape:
            expanded_state[key] = source_value
        elif key not in allowed_shape_changes:
            mismatched.append(
                f"{key}: checkpoint {tuple(source_value.shape)} != model "
                f"{tuple(target_state[key].shape)}"
            )
    if mismatched:
        raise ValueError("Unexpected warm-start shape mismatches:\n  " + "\n  ".join(mismatched))

    if input_key not in source_state or input_key not in target_state:
        raise ValueError("Feature-expanded warm start requires a linear input projection")
    source_input = source_state[input_key]
    target_input = expanded_state[input_key]
    if source_input.shape[0] != target_input.shape[0]:
        raise ValueError("Input projection output dimension changed during warm start")
    target_input.zero_()
    for source_index, feature_name in enumerate(source_features):
        target_index = target_features.index(feature_name)
        target_input[:, target_index] = source_input[:, source_index]
    if source_pid_count:
        source_start = len(source_features)
        target_start = len(target_features)
        target_input[:, target_start : target_start + target_pid_count] = source_input[
            :, source_start : source_start + source_pid_count
        ]
    source_trailing_start = len(source_features) + source_pid_count
    target_trailing_start = len(target_features) + target_pid_count
    source_trailing = source_input.shape[1] - source_trailing_start
    target_trailing = target_input.shape[1] - target_trailing_start
    if source_trailing != target_trailing:
        raise ValueError("Conditional input dimension changed during warm start")
    if source_trailing:
        target_input[:, target_trailing_start:] = source_input[:, source_trailing_start:]
    expanded_state[input_key] = target_input

    for key in (output_weight_key, output_bias_key):
        if key not in source_state or key not in target_state:
            raise ValueError("Feature-expanded warm start requires a linear output projection")
    source_output_weight = source_state[output_weight_key]
    source_output_bias = source_state[output_bias_key]
    target_output_weight = expanded_state[output_weight_key]
    target_output_bias = expanded_state[output_bias_key]
    if source_output_weight.shape[1] != target_output_weight.shape[1]:
        raise ValueError("Output projection input dimension changed during warm start")
    for source_index, feature_name in enumerate(source_features):
        target_index = target_features.index(feature_name)
        target_output_weight[target_index] = source_output_weight[source_index]
        target_output_bias[target_index] = source_output_bias[source_index]

    for feature_name, initializer in dict(
        expansion_cfg.get("output_initializers") or {}
    ).items():
        if feature_name not in target_features or feature_name in source_features:
            raise ValueError(f"Invalid new-feature output initializer for {feature_name!r}")
        copy_from = str(initializer["copy_from"])
        if copy_from not in source_features:
            raise ValueError(f"Initializer source feature {copy_from!r} is unavailable")
        source_index = source_features.index(copy_from)
        target_index = target_features.index(feature_name)
        target_output_weight[target_index] = source_output_weight[source_index]
        target_output_bias[target_index] = source_output_bias[source_index] + float(
            initializer.get("bias_offset", 0.0)
        )
    expanded_state[output_weight_key] = target_output_weight
    expanded_state[output_bias_key] = target_output_bias
    model.load_state_dict(expanded_state, strict=True)
    retained = sorted(set(target_state) - set(source_state))
    log.info(
        "Loaded feature-expanded warm start: source_features=%s target_features=%s "
        "retained_new_tensors=%s",
        source_features,
        target_features,
        retained,
    )


def _log_data_split_summary(trainer: L.Trainer, datamodule) -> None:
    if trainer.global_rank != 0 or not trainer.loggers:
        return
    if not hasattr(datamodule, "data_split_summary"):
        log.info("Datamodule does not expose data_split_summary; skipping split logging.")
        return

    summary = datamodule.data_split_summary()
    rows = summary.get("rows", [])
    if not rows:
        log.info("Datamodule split summary is empty; skipping split logging.")
        return

    log.info("Resolved data split summary:")
    for row in rows:
        log.info(
            "  split=%s suite=%s class=%s group=%s process=%s label=%s files=%s events=%s batch_size=%s "
            "sequence=%s eval_sequence=%s eval_min_pt=%s weight=%s",
            row.get("split"),
            row.get("suite"),
            row.get("class"),
            row.get("group"),
            row.get("process"),
            row.get("label"),
            row.get("file_count"),
            row.get("event_count"),
            row.get("batch_size"),
            row.get("sequence_type"),
            row.get("eval_sequence_type"),
            row.get("eval_min_pt"),
            row.get("sampling_weight"),
        )

    metrics = {}
    totals: dict[str, int] = {}
    for row in rows:
        split = row["split"]
        # Legacy data modules expose a class key. Canonical group-balanced
        # modules instead identify each row by suite, group, and process.
        # Keep every canonical test row distinct: tt appears in both test
        # suites and must not overwrite metrics from the other suite.
        class_name = row.get("class")
        if class_name is None:
            class_name = "/".join(
                str(value)
                for value in (row.get("suite"), row.get("group"), row.get("process"))
                if value is not None
            )
        if not class_name:
            class_name = "unlabelled"
        file_count = row.get("file_count")
        event_count = row.get("event_count")
        if file_count is not None:
            metrics[f"data_splits/{split}/{class_name}/file_count"] = int(file_count)
        if event_count is not None:
            event_count = int(event_count)
            metrics[f"data_splits/{split}/{class_name}/event_count"] = event_count
            totals[split] = totals.get(split, 0) + event_count
    for split, total in totals.items():
        metrics[f"data_splits/{split}/total_event_count"] = total

    columns = [
        "split",
        "suite",
        "class",
        "group",
        "process",
        "label",
        "file_count",
        "event_count",
        "batch_size",
        "sequence_type",
        "min_pt",
        "eval_sequence_type",
        "eval_min_pt",
        "sampling_weight",
    ]
    table_data = [[row.get(column) for column in columns] for row in rows]

    for lightning_logger in trainer.loggers:
        lightning_logger.log_hyperparams({"data_splits": summary})
        if metrics:
            lightning_logger.log_metrics(metrics)
        if isinstance(lightning_logger, L.pytorch.loggers.WandbLogger):
            try:
                import wandb

                lightning_logger.experiment.config.update(
                    {"data_splits": summary},
                    allow_val_change=True,
                )
                lightning_logger.experiment.log(
                    {"data_splits/table": wandb.Table(columns=columns, data=table_data)},
                    commit=False,
                )
            except Exception as exc:
                log.warning(f"Failed to log data split table to W&B: {exc}")


def _log_full_config_to_wandb(
    trainer: L.Trainer,
    cfg: DictConfig,
    cfg_path: str | Path,
    cfg_resolved_path: str | Path,
) -> None:
    """Upload complete Hydra configs to W&B without replacing existing hparams."""
    if trainer.global_rank != 0 or not trainer.loggers:
        return

    cfg_path = Path(cfg_path)
    cfg_resolved_path = Path(cfg_resolved_path)
    for lightning_logger in trainer.loggers:
        if not isinstance(lightning_logger, L.pytorch.loggers.WandbLogger):
            continue
        try:
            import wandb

            run = lightning_logger.experiment
            config_payload = {
                "full_config": OmegaConf.to_container(
                    cfg,
                    resolve=False,
                    throw_on_missing=False,
                ),
                "full_config_resolved": OmegaConf.to_container(
                    cfg,
                    resolve=True,
                    throw_on_missing=False,
                ),
                "full_config_files": {
                    "config": str(cfg_path),
                    "config_resolved": str(cfg_resolved_path),
                },
            }
            run.config.update(config_payload, allow_val_change=True)

            artifact = wandb.Artifact(
                name=f"{run.id}-full-config",
                type="config",
                description="Complete unresolved and resolved Hydra configs for this run.",
            )
            if cfg_path.is_file():
                artifact.add_file(str(cfg_path), name="config.yaml")
            if cfg_resolved_path.is_file():
                artifact.add_file(str(cfg_resolved_path), name="config_resolved.yaml")
            run.log_artifact(artifact)
            log.info("Uploaded full Hydra config to W&B config and artifact.")
        except Exception as exc:
            log.warning(f"Failed to upload full config to W&B: {exc}")


def get_nodename_bigram():
    """Generate a unique run identifier based on the nodename and a random bigram.
    Example: `max-wng029_QuickBear`

    If a job ID exists (this needs to be passed to the container in the job
    submission script via eg. --env JOB_ID="$SLURM_JOB_ID"), the bigram will be
    seeded based on this ID. This means that a multi-node run will have the same
    bigram, which is useful when the processes on one node need to access files
    in the directory belonging to the main node containing rank 0.

    If no job ID exists, the bigram is generated from the nodename and the
    current time, which means that two runs starting at the same time on
    different nodes will have different bigrams (if the nodename is not included,
    two runs starting at the same time will have the same bigram).

    Returns:
        str: Unique run identifier.
    """
    nodename = os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "local")
    job_id = os.environ.get("JOB_ID", None)

    # cleanup
    nodename = nodename.split(".")[0]

    nodename_with_time = f"{nodename}_{int(time.time())}"

    # get hashes
    if job_id is not None:
        log.info(f"Job ID {job_id} detected. Generating bigram based on this job ID.")
        hashed_name = hashlib.sha256(job_id.encode()).hexdigest()
    else:
        log.info(
            f"No job ID detected. Generating bigram based on node name and time stamp, {nodename_with_time}."
        )
        hashed_name = hashlib.sha256(nodename_with_time.encode()).hexdigest()

    # bigram
    bigram = get_bigram(seed=int(hashed_name, 16))

    if job_id is not None:
        return "_".join([nodename, bigram, job_id])
    else:
        return "_".join([nodename, bigram])


# this is how we can include this resolver in the run directory (see configs/hydra/default.yaml)
OmegaConf.register_new_resolver("nodename_bigram", get_nodename_bigram, use_cache=True)
# add eval resolver to evaluate expressions in the config
OmegaConf.register_new_resolver("eval", eval)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # check if cuda available
    if not torch.cuda.is_available():
        log.warning("CUDA is not available!")
    else:
        log.info("CUDA is available.")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Git Status: {git_utils.get_git_status()}")
    log.info(f"Git Hash: {git_utils.get_git_hash()}")
    log.info(f"Last Commit Message: {git_utils.get_last_commit_message()}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    if cfg.get("continue_from_checkpoint", False):
        if cfg.continue_from_checkpoint is None:
            is_checkpoint_continuation_run = False
        elif isinstance(cfg.continue_from_checkpoint, str):
            is_checkpoint_continuation_run = True
        else:
            raise ValueError(
                "continue_from_checkpoint should be either a string, `False` or `None`"
                f", but got {cfg.continue_from_checkpoint}"
            )
    else:
        is_checkpoint_continuation_run = False

    if is_checkpoint_continuation_run:
        ckpt_path = Path(cfg.continue_from_checkpoint)
        log.info(f"Loading model from lightning checkpoint {ckpt_path}")
        resolved_cfg_path = ckpt_path.parent.parent / "config_resolved.yaml"
        log.info(f"Loading config from {resolved_cfg_path}")
        cfg_ckpt = OmegaConf.load(resolved_cfg_path)
        cfg = cfg_ckpt
        model_class_name = cfg.model._target_.split(".")[-1]
        try:
            lightning_module_class = getattr(gabbro_lightning_models, model_class_name)
        except AttributeError:
            raise AttributeError(
                f"Model class {model_class_name} not found in gabbro.lightning_models. "
                f"To be able to load a model from a checkpoint, the model class must be "
                f"imported in gabbro.lightning_models. "
                f"Available models are: {dir(gabbro_lightning_models)}"
            )
        model = lightning_module_class.load_from_checkpoint(ckpt_path, weights_only=False)
    else:
        ckpt_path = None
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: L.LightningModule = hydra.utils.instantiate(cfg.model)

        if cfg.get("load_weights_from", False):
            log.info(f"Loading model weights from {cfg.load_weights_from}")

            load_cpt_path = Path(cfg.load_weights_from).parent.parent / "config.yaml"
            print("Model config before loading weights:")
            print(OmegaConf.to_yaml(cfg.model))
            cfg_ckpt = OmegaConf.load(load_cpt_path)
            OmegaConf.update(
                cfg.model,
                "model_kwargs_loaded",
                cfg_ckpt.model.model_kwargs,
                force_add=True,
            )

            # we want to only load the weights, not the optimizer state etc. as would
            # be done with LightningModule.load_from_checkpoint()
            state_dict = torch.load(
                cfg.load_weights_from,
                map_location="cpu",
                weights_only=False,
            )["state_dict"]  # nosec
            expansion_cfg = cfg.get("load_weights_feature_expansion") or {}
            if expansion_cfg.get("enabled", False):
                _load_feature_expanded_weights(
                    model,
                    state_dict,
                    cfg_ckpt,
                    cfg,
                    expansion_cfg,
                )
            else:
                model.load_state_dict(state_dict, strict=cfg.get("load_weights_strict", True))

            log.info("Model config after loading weights:")
            log.info(OmegaConf.to_yaml(cfg.model))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # get the experiment_key from the comet logger
    for logger_i in logger:
        if isinstance(logger_i, L.pytorch.loggers.CometLogger):
            experiment_key = logger_i.experiment.get_key()
            log.info(f"Comet experiment_key: {experiment_key}")
            cfg.logger.comet.experiment_key = experiment_key

    log.info("Instantiating callbacks...")

    callbacks: Dict[str, L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Done instantiating callbacks.")
    log.info("Callbacks:")
    for cb_name, cb in callbacks.items():
        log.info(f"- {cb_name}: {cb}")

    log.info(f"Model: \n{model}")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=list(callbacks.values()),
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "git": {
            "git_hash": git_utils.get_git_hash(),
            "git_status": git_utils.get_git_status(),
            "git_last_commit_message": git_utils.get_last_commit_message(),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", None),
            "log_file": os.environ.get("SLURM_LOGFILE", None),
        },
        "load_weights_from": cfg.get("load_weights_from", None),
        "gpu_properties": get_gpu_properties(),
    }
    metric_dict = {}

    log.info(f"Slurm job ID: {object_dict['slurm']['job_id']}")

    if logger and cfg.get("ckpt_path_for_evaluation") is None:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        _log_data_split_summary(trainer, datamodule)

    if cfg.get("train"):
        # --- Save config for reproducibility --- #
        # save config
        cfg_backup_file = f"{cfg.trainer.get('default_root_dir')}/config.yaml"
        if not is_checkpoint_continuation_run and trainer.global_rank == 0:
            with open(cfg_backup_file, "w") as f:
                log.info(f"Saving config to {cfg_backup_file}")
                OmegaConf.save(cfg, f)
            # save resolved config
            cfg_resolved_file = f"{cfg.trainer.get('default_root_dir')}/config_resolved.yaml"
            with open(cfg_resolved_file, "w") as f:
                log.info(f"Saving resolved config to {cfg_resolved_file}")
                OmegaConf.save(cfg, f, resolve=True)
            _log_full_config_to_wandb(
                trainer,
                cfg,
                cfg_backup_file,
                cfg_resolved_file,
            )
        # ---

        log.info("------------------")
        log.info("Starting training!")
        log.info("------------------")
        log.info(f"Global rank: {trainer.global_rank}")
        # saving the checkpoint of the untrained model
        if not is_checkpoint_continuation_run:
            # save state dict of the untrained model
            untrained_model_checkpoint = (
                f"{cfg.trainer.default_root_dir}/untrained_model_state_dict.ckpt"
            )
            # make directory
            Path(untrained_model_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), untrained_model_checkpoint)
            log.info(f"Saved untrained model state dict to {untrained_model_checkpoint}")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        metric_dict.update(dict(trainer.callback_metrics))

    if cfg.get("test"):
        log.info("-----------------")
        log.info("Starting testing!")
        log.info("-----------------")

        if cfg.get("test_without_checkpoint", False):
            log.info("`test_without_checkpoint` is enabled; testing current model state.")
            ckpt_path = None
            process_rank = get_rank() if torch.distributed.is_initialized() else 0
        elif cfg.get("ckpt_path_for_evaluation") is not None:
            # evaluate a specific checkpoint
            ckpt_path = cfg.get("ckpt_path_for_evaluation")
            process_rank = get_rank() if torch.distributed.is_initialized() else 0
        else:
            # No specific checkpoint for evaluation was provided --> this is what happens
            # at the end of training when the model is tested on the best weights
            # --> we need to find the best ckpt in the callbacks. This may be in the
            # directory of another node (it only saves checkpoints on rank 0).

            # Keep track of which process gets which checkpoint
            process_rank = get_rank() if torch.distributed.is_initialized() else 0
            world_size = get_world_size() if torch.distributed.is_initialized() else 1

            # Check if the node's root directory contains checkpoints at all
            if os.path.isdir(f"{cfg.trainer.default_root_dir}/checkpoints"):
                name_best_ckpt = "model_checkpoint_best"
                name_ckpt = "model_checkpoint"
                if name_best_ckpt in callbacks:
                    ckpt_path = callbacks[name_best_ckpt].best_model_path
                    log.info(
                        f"Rank {process_rank}/[0-{world_size - 1}]: Using best model path from callback {name_best_ckpt}: {ckpt_path}"
                    )
                    if ckpt_path == "":
                        ckpt_path = f"{cfg.trainer.default_root_dir}/checkpoints/last.ckpt"
                        log.warning(
                            f"Rank {process_rank}/[0-{world_size - 1}]: Callback "
                            f"{name_best_ckpt} did not report a best checkpoint. "
                            f"Falling back to {ckpt_path}"
                        )
                # if best ckpt not found in that callback, try with other name
                elif name_ckpt in callbacks:
                    ckpt_path = callbacks[name_ckpt].best_model_path
                    log.info(
                        f"Rank {process_rank}/[0-{world_size - 1}]: Using best model path from callback {name_ckpt}: {ckpt_path}"
                    )
                    if ckpt_path == "":
                        ckpt_path = f"{cfg.trainer.default_root_dir}/checkpoints/last.ckpt"
                        log.warning(
                            f"Rank {process_rank}/[0-{world_size - 1}]: Callback {name_ckpt} "
                            f"did not report a best checkpoint. Falling back to {ckpt_path}"
                        )
                # if best ckpt not found there either, just use the last ckpt
                # which is stored separately as last.ckpt
                else:
                    log.warning(
                        f"Neither '{name_best_ckpt}' nor '{name_ckpt}' found in callbacks!"
                    )
                    log.warning(
                        f"Rank {process_rank}/[0-{world_size - 1}]: Best ckpt not found! Using last.ckpt for testing..."
                    )
                    ckpt_path = f"{cfg.trainer.default_root_dir}/checkpoints/last.ckpt"
            else:
                # If the root directory does not have checkpoints, look for one that does
                root_dir = cfg.trainer.default_root_dir
                log.info(
                    f"Rank {process_rank}/[0-{world_size - 1}]: The root directory {root_dir} does not contain checkpoints. Searching other directories..."
                )
                bigram = (root_dir.split("/")[-1]).split("_")[-2]
                log.info(f"Extracted bigram {bigram}")
                # Get a list of all paths with this bigram in the parent directory of current run directory
                p = Path(root_dir)
                p = p.resolve()
                directory_list = glob.glob(f"{os.path.join(p.parent, '*' + str(bigram) + '*')}")
                directory_list.sort()
                log.info(f"List of directories containing the bigram {bigram}: {directory_list}")
                found_checkpoints = False
                for directory in directory_list:
                    # Check if it contains a checkpoint directory
                    if os.path.isdir(f"{directory}/checkpoints"):
                        checkpoint_directory = f"{directory}/checkpoints"
                        log.info(f"Checkpoint directory detected: {checkpoint_directory}")
                        # Now try to find the checkpoint
                        # -- best.ckpt
                        if os.path.isfile(f"{checkpoint_directory}/best.ckpt"):
                            ckpt_path = f"{checkpoint_directory}/best.ckpt"
                            log.info(
                                f"Rank {process_rank}/[0-{world_size - 1}]: Using best checkpoint from {ckpt_path}"
                            )
                        # -- last.ckpt
                        elif os.path.isfile(f"{checkpoint_directory}/last.ckpt"):
                            ckpt_path = f"{checkpoint_directory}/last.ckpt"
                            log.info(
                                f"Rank {process_rank}/[0-{world_size - 1}]: Using last checkpoint from {ckpt_path}"
                            )
                        # -- the very last .ckpt file in the list of all checkpoint files
                        else:
                            all_checkpoints = glob.glob(
                                f"{os.path.join(checkpoint_directory, '*.ckpt')}"
                            )
                            # If the directory does not have any .ckpt files
                            if len(all_checkpoints) == 0:
                                log.info(f"Can not find any checkpoints in {checkpoint_directory}")
                                continue
                            all_checkpoints.sort()
                            ckpt_path = all_checkpoints[-1]  # Take the last one
                            log.info(
                                f"Rank {process_rank}/[0-{world_size - 1}]: Using last checkpoint from {ckpt_path}"
                            )
                        found_checkpoints = True
                # if we still couldn't find the checkpoint
                assert found_checkpoints, (
                    f"Rank {process_rank}/[0-{world_size - 1}]: No checkpoints could be found, exiting."
                )
        # ------------------------------------------------

        # update the default root dir for testing
        ckpt_filename = Path(ckpt_path).name if ckpt_path else "current"
        evaluation_output_name = cfg.get("evaluation_output_name") or ckpt_filename
        evaluation_output_name = str(evaluation_output_name)
        if (
            Path(evaluation_output_name).name != evaluation_output_name
            or evaluation_output_name in {".", ".."}
        ):
            raise ValueError(
                "evaluation_output_name must be a single directory name, got "
                f"{evaluation_output_name!r}"
            )
        cfg.trainer.default_root_dir = (
            Path(cfg.trainer.default_root_dir) / "evaluation" / evaluation_output_name
        )
        cfg.trainer.default_root_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Set default_root_dir to {trainer.default_root_dir}")

        if ckpt_path == "":
            log.warning(
                "Best ckpt either not found or not accessible in the callbacks! "
                "Using current weights for testing..."
            )
        log.info(f"Best ckpt path: {ckpt_path}")

        log.info(f"Instantiating trainer for testing <{cfg.trainer._target_}>")

        if cfg.get("test_only_on_rank_zero", False):
            log.info("`test_only_on_rank_zero` is enabled")
            # for testing, reinitialized the datamodule and trainer with only one rank
            # (this ensures that cases where callbacks only consider rank 0 work correctly
            # in terms of using as much data as we want it to use)
            if torch.distributed.is_initialized():
                log.info("Destroying distributed process group.")
                torch.distributed.destroy_process_group()
            if process_rank == 0:
                log.info(f"Reinitializing datamodule and trainer for rank {process_rank}")
                datamodule = hydra.utils.instantiate(cfg.data)
                cfg.trainer.num_nodes = 1
                cfg.trainer.devices = 1
                cfg.trainer.strategy = "auto"
                trainer = hydra.utils.instantiate(
                    cfg.trainer,
                    logger=logger if cfg.get("ckpt_path_for_evaluation") is None else None,
                    callbacks=list(callbacks.values()),
                )
                trainer.test(
                    model=model,
                    datamodule=datamodule,
                    ckpt_path=ckpt_path,
                    weights_only=False,
                )
            else:
                log.info(f"Skipping testing on rank {process_rank}")
        else:
            trainer = hydra.utils.instantiate(
                cfg.trainer,
                logger=logger if cfg.get("ckpt_path_for_evaluation") is None else None,
                callbacks=list(callbacks.values()),
            )
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                weights_only=False,
            )
        metric_dict.update(dict(trainer.callback_metrics))

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # set CUDA_LAUNCH_BLOCKING=1 to get more informative stack traces
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.set_float32_matmul_precision("medium")

    log.info(f"output_dir: {cfg.paths.output_dir}")
    # check if the output directory for auto-resubmission is set
    if cfg.get("output_dir_resub") is not None:
        output_dir_resub = Path(cfg.output_dir_resub)
        log.info(f"Setting output_dir to {output_dir_resub}")
        log.info(f"Creating output_dir {output_dir_resub}")
        output_dir_resub.mkdir(parents=True, exist_ok=True)
        cfg.paths.output_dir = output_dir_resub

        # check if checkpoint `last.ckpt` exists in the output directory /checkpoints
        last_ckpt = output_dir_resub / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            log.info(f"Found last.ckpt in {last_ckpt}. Will continue from this checkpoint.")
            log.info(f"Setting continue_from_checkpoint={last_ckpt}")
            cfg.continue_from_checkpoint = str(last_ckpt)

    experiment_name = cfg.get("task_name") or Path(cfg.trainer.default_root_dir).name
    if cfg.logger.get("comet") is not None:
        if not cfg.logger.comet.get("experiment_name"):
            cfg.logger.comet.experiment_name = experiment_name
    if cfg.logger.get("wandb") is not None:
        if not cfg.logger.wandb.get("name"):
            cfg.logger.wandb.name = experiment_name

    # load full config from file if specified
    if cfg.get("ckpt_path_for_evaluation") is not None:
        ckpt_path = Path(cfg.ckpt_path_for_evaluation)
        log.info(f"Will evaluate the model checkpoint {ckpt_path}")

        cfg_ckpt_path = ckpt_path.parent.parent / "config.yaml"
        log.info(f"Loading config from {cfg_ckpt_path} for evaluation")
        cfg_ckpt = OmegaConf.load(cfg_ckpt_path)

        # get the redundant output dir created by hydra when this evaluation
        # was started (hydra always creates a new output dir for each execution
        # but we want to use the same output dir as the training run)
        redundant_output_dir = Path(cfg.paths.output_dir)
        # compose cfg from the new cfg_path + the overrides in the
        # redundant output dir / ".hydra" / "overrides.yaml" (cause those are the
        # ones passed in the command line)
        ConfigStore.instance().store("cfg_ckpt", node=cfg_ckpt)
        checkpoint_overrides = [
            override
            for override in HydraConfig.get().overrides.task
            # The initial Hydra composition may need an experiment selector,
            # but the loaded checkpoint is already a complete config and has
            # no experiment defaults entry to override.
            if not override.startswith("experiment=")
        ]
        cfg = compose(config_name="cfg_ckpt", overrides=checkpoint_overrides)

        # A checkpoint normally inherits its training datamodule verbatim.  For
        # cross-domain tests, merge only the data section from another config
        # while retaining the checkpoint's model, feature, and loss settings.
        evaluation_data_config = cfg.get("evaluation_data_config")
        if evaluation_data_config:
            evaluation_data_path = Path(str(evaluation_data_config)).expanduser()
            if not evaluation_data_path.is_absolute():
                evaluation_data_path = Path.cwd() / evaluation_data_path
            if not evaluation_data_path.is_file():
                raise FileNotFoundError(
                    f"Evaluation data config does not exist: {evaluation_data_path}"
                )
            log.info(f"Loading evaluation data from {evaluation_data_path}")
            evaluation_cfg = OmegaConf.load(evaluation_data_path)
            evaluation_data = evaluation_cfg.get("data", evaluation_cfg)
            # These mappings describe a complete dataset.  Replace rather than
            # recursively merge them so processes from the training domain do
            # not leak into the evaluation domain.
            replacement_keys = {
                "process_catalog",
                "train_val_processes",
                "test_suites",
            }
            evaluation_data_values = OmegaConf.to_container(
                evaluation_data,
                resolve=False,
            )
            merge_values = {
                key: value
                for key, value in evaluation_data_values.items()
                if key not in replacement_keys
            }
            merged_data = OmegaConf.merge(cfg.data, merge_values)
            OmegaConf.set_struct(merged_data, False)
            for key in replacement_keys:
                if key in evaluation_data_values:
                    merged_data[key] = evaluation_data_values[key]
            cfg.data = merged_data

        selected_test_suites = cfg.get("evaluation_test_suites")
        if selected_test_suites:
            if isinstance(selected_test_suites, str):
                selected_test_suites = [selected_test_suites]
            available_test_suites = cfg.data.get("test_suites") or {}
            missing_test_suites = [
                name for name in selected_test_suites if name not in available_test_suites
            ]
            if missing_test_suites:
                raise ValueError(
                    "Unknown evaluation test suite(s): "
                    + ", ".join(map(str, missing_test_suites))
                )
            cfg.data.test_suites = {
                name: available_test_suites[name] for name in selected_test_suites
            }

        # set the output dir to the parent of the ckpt config path
        log.info(f"Setting output dir to {cfg_ckpt_path.parent}")
        cfg.paths.output_dir = cfg_ckpt_path.parent

        remove_empty_hydra_run_dir(redundant_output_dir)

        log.info(f"paths.output_dir={cfg.paths.output_dir}")
        log.info(
            "logger.comet.experiment_name=%s",
            OmegaConf.select(cfg, "logger.comet.experiment_name", default=None),
        )

        # set the evaluation flag to True and the training flag to False
        log.info("Setting evaluation flag to True and training flag to False")
        cfg.train = False
        cfg.test = True
        # set to single-node, single-gpu strategy for evaluation (because this will
        # crash otherwise if the model is evaluated on a single GPU and was trained
        # on multiple GPUs)
        # log.info("Setting trainer to single-node, single-gpu strategy for evaluation")
        # cfg.trainer.num_nodes = 1
        # cfg.trainer.devices = 1
        # cfg.trainer.strategy = "auto"

    # train the model
    metric_dict, _ = train(cfg)

    return get_metric_value(metric_dict, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
