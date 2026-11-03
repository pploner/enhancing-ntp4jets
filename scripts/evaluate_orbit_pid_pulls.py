#!/usr/bin/env python
"""Plot same-token reconstruction residuals conditioned on the original PID.

The decoder does not predict per-particle uncertainties, so these distributions
are dimensionless/angular residuals rather than statistical pulls of the form
``(reconstructed - original) / uncertainty``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_explicit_env_file() -> None:
    env_file = os.environ.get("GABBRO_ENV_FILE")
    if not env_file:
        return
    env_path = Path(os.path.expandvars(os.path.expanduser(env_file)))
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
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ[key] = os.path.expandvars(value)


_load_explicit_env_file()
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from gabbro.data.orbit_parquet import PID_CLASS_NAMES, SEQUENCE_SCHEMAS, OrbitParquetDataset
from gabbro.plotting.orbit import (
    close_figure,
    pid_residual_arrays,
    pid_residual_distribution_summary,
    plot_pid_conditional_residuals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--test-manifest", type=Path)
    source.add_argument(
        "--test-class",
        action="append",
        metavar="CLASS=MANIFEST[,MANIFEST...]",
        help="Repeat for balanced top-level classes; manifests within a class are balanced.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Log the plot and table to W&B.")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Upload existing output-dir artifacts without rerunning inference.",
    )
    parser.add_argument("--wandb-project", default="orbit-tokenizer")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group", default="orbit-pid-conditional-residuals")
    parser.add_argument("--wandb-entity")
    return parser.parse_args()


def parse_test_classes(values: list[str]) -> dict[str, list[Path]]:
    """Parse balanced CLASS=manifest[,manifest...] command-line entries."""
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CLASS=MANIFEST[,MANIFEST...], got {value!r}")
        name, manifests = value.split("=", 1)
        paths = [Path(item).resolve() for item in manifests.split(",") if item]
        if not name or not paths or any(not path.is_file() for path in paths):
            raise ValueError(f"Invalid class specification: {value!r}")
        result[name] = paths
    return result


def split_event_quota(total: int, count: int) -> list[int]:
    """Split an integer event budget deterministically and without loss."""
    return [total // count + (index < total % count) for index in range(count)]


def load_model(run_dir: Path, device: torch.device):
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        config_path = run_dir / ".hydra" / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(f"No resolved or Hydra config found under {run_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cfg = OmegaConf.load(config_path)
    if not bool(cfg.get("pid", {}).get("enabled", False)):
        raise ValueError("PID-conditioned evaluation requires a PID-enabled checkpoint")
    model = instantiate(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), cfg, checkpoint_path


def checkpoint_data_settings(cfg) -> tuple[str, int, str | None, float]:
    sequence_type = str(cfg.data.get("sequence_type", "particle"))
    if sequence_type not in SEQUENCE_SCHEMAS or not sequence_type.startswith("particle"):
        raise ValueError(f"Expected a particle checkpoint, got {sequence_type!r}")
    schema = SEQUENCE_SCHEMAS[sequence_type]
    max_length = int(cfg.data.get("max_sequence_length") or schema["max_sequence_length"])
    mask_column = cfg.data.get("mask_column")
    if mask_column is None:
        mask_column = schema["mask_column"]
    mask_min_value = cfg.data.get("mask_min_value")
    if mask_min_value is None:
        mask_min_value = schema["mask_min_value"]
    return sequence_type, max_length, mask_column, float(mask_min_value)


def to_physical(values: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            values[..., 0] * 3.0,
            np.arctan2(values[..., 2], values[..., 1]),
            np.clip(np.exp(values[..., 3] + 1.8) - 1e-8, 0.0, None),
        ),
        axis=-1,
    )


def decode_batch(model, batch, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    features = batch["part_features"].to(device)
    mask = batch["part_mask"].to(device)
    pid = batch["part_pid"].to(device)
    with torch.inference_mode():
        reconstructed, outputs = model.forward(features, mask, pid_particle=pid)
    pid_logits = outputs.get("pid_logits")
    if pid_logits is None:
        raise ValueError("PID-enabled checkpoint did not return decoded PID logits")
    return (
        reconstructed.detach().cpu().numpy(),
        pid_logits.argmax(dim=-1).detach().cpu().numpy(),
    )


def plot_pulls(by_pid: dict[int, dict[str, np.ndarray]], output_path: Path, bins: int) -> None:
    figure = plot_pid_conditional_residuals(by_pid, PID_CLASS_NAMES, bins=bins)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    close_figure(figure)


def upload_to_wandb(args: argparse.Namespace) -> str:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B upload requested, but wandb is not installed") from exc

    summary_path = args.output_dir / "pid_conditional_pull_summary.json"
    plot_path = args.output_dir / "pid_conditional_pull_histograms.png"
    values_path = args.output_dir / "pid_conditional_pull_values.npz"
    for path in (summary_path, plot_path, values_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing PID-residual artifact: {path}")
    summary = json.loads(summary_path.read_text())
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.output_dir.name,
        group=args.wandb_group,
        job_type="evaluation",
        config={
            "events": summary["events"],
            "particles": summary["particles"],
            "run_dir": summary["run_dir"],
            "checkpoint": summary["checkpoint"],
            "test_manifests": summary.get(
                "test_manifests", {"sample": [summary.get("test_manifest")]}
            ),
            "conditioning": summary["conditioning"],
        },
    )
    try:
        first_class = summary["classes"][PID_CLASS_NAMES[0]]
        residual_keys = [
            key for key, value in first_class.items()
            if isinstance(value, dict) and "rmse" in value
        ]
        columns = ["PID", "count", "PID accuracy"] + [
            f"{key} {stat}" for key in residual_keys for stat in ("bias", "RMSE")
        ]
        rows = []
        for class_name in PID_CLASS_NAMES:
            values = summary["classes"][class_name]
            rows.append(
                [
                    class_name,
                    values["count"],
                    values["pid_accuracy"],
                    *[
                        values[key][stat.lower()]
                        for key in residual_keys
                        for stat in ("bias", "RMSE")
                    ],
                ]
            )
        run.log(
            {
                "pid_conditional/residual_histograms": wandb.Image(str(plot_path)),
                "pid_conditional/summary": wandb.Table(columns=columns, data=rows),
                "pid_conditional/pid_accuracy": summary["pid_accuracy"],
            }
        )
        artifact = wandb.Artifact(
            name=f"pid-conditional-residuals-{run.id}",
            type="orbit-downstream-evaluation",
            metadata={"events": summary["events"], "particles": summary["particles"]},
        )
        for path in (plot_path, summary_path, values_path):
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        return run.url
    finally:
        wandb.finish()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.upload_only:
        if not args.wandb:
            raise ValueError("--upload-only requires --wandb")
        print(f"Uploaded PID-conditioned residuals to {upload_to_wandb(args)}")
        return
    if args.run_dir is None or (args.test_manifest is None and not args.test_class):
        raise ValueError(
            "--run-dir and either --test-manifest or --test-class are required "
            "unless --upload-only is used"
        )
    if args.events < 1 or args.bins < 1:
        raise ValueError("--events and --bins must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, checkpoint_path = load_model(args.run_dir.resolve(), device)
    sequence_type, max_length, mask_column, mask_min_value = checkpoint_data_settings(cfg)
    batch_size = args.batch_size or (16 if max_length > 128 else 256)
    if args.test_class:
        test_classes = parse_test_classes(args.test_class)
    else:
        test_classes = {"sample": [args.test_manifest.resolve()]}
    datasets = []
    for (class_name, manifests), class_quota in zip(
        test_classes.items(), split_event_quota(args.events, len(test_classes)), strict=True
    ):
        for manifest, manifest_quota in zip(
            manifests, split_event_quota(class_quota, len(manifests)), strict=True
        ):
            datasets.append(
                (
                    class_name,
                    OrbitParquetDataset(
                        manifest,
                        sequence_type=sequence_type,
                        max_sequence_length=max_length,
                        batch_size=batch_size,
                        mask_column=mask_column,
                        mask_min_value=mask_min_value,
                        max_events=manifest_quota,
                        pid_cfg=cfg.get("pid"),
                        include_energy=bool(cfg.data.get("include_energy", False)),
                        energy_shift=float(cfg.data.get("energy_shift", 2.5)),
                    ),
                )
            )

    original_chunks = []
    reconstructed_chunks = []
    original_pid_chunks = []
    decoded_pid_chunks = []
    processed_events = 0
    with tqdm(total=args.events, desc="Evaluating PID-conditioned residuals", unit="event") as bar:
        for _, dataset in datasets:
            for batch in dataset:
                reconstructed, decoded_pid = decode_batch(model, batch, device)
                original = batch["part_features"].numpy()
                original_pid = batch["part_pid"].numpy()
                mask = batch["part_mask"].numpy().astype(bool)
                original_chunks.append(original[mask])
                reconstructed_chunks.append(reconstructed[mask])
                original_pid_chunks.append(original_pid[mask])
                decoded_pid_chunks.append(decoded_pid[mask])
                batch_events = len(mask)
                processed_events += batch_events
                bar.update(batch_events)

    original = np.concatenate(original_chunks)
    reconstructed = np.concatenate(reconstructed_chunks)
    original_pid = np.concatenate(original_pid_chunks)
    decoded_pid = np.concatenate(decoded_pid_chunks)
    feature_names = list(cfg.get("feature_dict", {}).keys())
    all_residuals = pid_residual_arrays(original, reconstructed, feature_names)
    by_pid = {
        pid: {key: values[original_pid == pid] for key, values in all_residuals.items()}
        for pid in range(len(PID_CLASS_NAMES))
    }

    plot_path = args.output_dir / "pid_conditional_pull_histograms.png"
    plot_pulls(by_pid, plot_path, args.bins)
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "test_manifests": {
            name: [str(path) for path in paths] for name, paths in test_classes.items()
        },
        "events": processed_events,
        "particles": int(len(original_pid)),
        "conditioning": "original PID at the same sequence position",
        "note": "Residuals are not uncertainty-normalized statistical pulls.",
        "pid_accuracy": float(np.mean(decoded_pid == original_pid)),
        "classes": {
            class_name: {
                "count": int(np.sum(original_pid == pid)),
                "pid_accuracy": float(np.mean(decoded_pid[original_pid == pid] == pid))
                if np.any(original_pid == pid)
                else None,
                **{
                    key: pid_residual_distribution_summary(values)
                    for key, values in by_pid[pid].items()
                },
            }
            for pid, class_name in enumerate(PID_CLASS_NAMES)
        },
    }
    (args.output_dir / "pid_conditional_pull_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "pid_conditional_pull_values.npz",
        original_pid=original_pid,
        decoded_pid=decoded_pid,
        **all_residuals,
    )
    print(f"Saved {plot_path}")
    if args.wandb:
        print(f"Uploaded PID-conditioned residuals to {upload_to_wandb(args)}")


if __name__ == "__main__":
    main()
