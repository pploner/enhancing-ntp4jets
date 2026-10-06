"""Composition checks for the portable VQ-STE SM+DY LR sweep configs."""

import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
ALPHAS = (1, 5, 10)
LEARNING_RATES = ("1e-7", "1e-6", "1e-5", "1e-4", "1e-3")
CONFIG_ROOT = "vq_ste_4096_sm_dy_lr_sweeps"


def _selector(alpha: int, learning_rate: str) -> str:
    return f"{CONFIG_ROOT}/alpha_{alpha}/lr_{learning_rate}"


def _physics_signature(cfg) -> dict:
    signature = copy.deepcopy(OmegaConf.to_container(cfg, resolve=False))
    signature["project_name"] = "<scan>"
    signature["task_name"] = "<scan>"
    signature["tags"] = ["<scan>"]
    signature["model"]["model_kwargs"]["alpha"] = "<alpha>"
    signature["model"]["optimizer"]["lr"] = "<lr>"
    return signature


def test_all_vq_ste_sm_dy_lr_sweep_configs_compose(monkeypatch):
    monkeypatch.setenv(
        "ORBIT_MANIFEST_DIR",
        str(REPO_ROOT / "manifests" / "production_final"),
    )
    signatures = []
    names = set()

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        for alpha in ALPHAS:
            for learning_rate in LEARNING_RATES:
                cfg = compose(
                    config_name="train.yaml",
                    overrides=[f"experiment={_selector(alpha, learning_rate)}"],
                )

                assert cfg.model.model_kwargs.alpha == alpha
                assert cfg.model.optimizer.lr == float(learning_rate)
                assert cfg.model.scheduler is None
                assert cfg.model.model_kwargs.vq_kwargs.num_codes == 4096
                assert cfg.model.model_kwargs.vq_kwargs.rotation_tr is False

                assert cfg.data.sequence_type == "particle_full"
                assert cfg.data.max_sequence_length == 500
                assert cfg.data.mask_column is None
                assert cfg.data.mask_min_value is None
                assert cfg.data.train_event_budget == 200_000
                assert cfg.data.val_event_budget == 20_000
                assert cfg.data.batch_size == 256
                assert cfg.trainer.accumulate_grad_batches == 1

                groups = {
                    process.group
                    for process in cfg.data.train_val_processes.values()
                }
                assert groups == {"QCD", "tt", "VJets", "VV", "DY"}
                assert cfg.data.train_val_processes.DYJetsToLL.group == "DY"

                assert cfg.trainer.max_epochs == 30
                assert cfg.callbacks.early_stopping.monitor == "val_loss"
                assert cfg.callbacks.early_stopping.patience == 4
                assert cfg.train is True
                assert cfg.test is False

                assert cfg.task_name not in names
                names.add(cfg.task_name)
                assert f"alpha-{alpha}" in cfg.tags
                assert f"lr-{learning_rate}" in cfg.tags
                signatures.append(_physics_signature(cfg))

    assert len(names) == 15
    assert all(signature == signatures[0] for signature in signatures[1:])
