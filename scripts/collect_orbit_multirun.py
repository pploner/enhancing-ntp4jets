#!/usr/bin/env python
"""Aggregate ORBIT-style artifacts produced by ORBIT training jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = dict
    OmegaConf = None

from gabbro.plotting.orbit import (
    PRESENTATION_CODEBOOK_MARKER_AREA_SCALE,
    multirun_color,
    plot_multirun_feature_histograms,
    plot_multirun_metric,
    plot_multirun_residual_histograms,
)


def _load_explicit_env_file() -> None:
    """Load ``GABBRO_ENV_FILE`` exactly like the training entrypoint does."""
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
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = os.path.expandvars(value)


_load_explicit_env_file()


def _load_config(path: Path) -> Any:
    if OmegaConf is not None:
        return OmegaConf.load(path)
    with path.open() as file:
        return yaml.safe_load(file)


def _select(cfg: Any, path: str, default: Any = None) -> Any:
    if OmegaConf is not None:
        return OmegaConf.select(cfg, path, default=default)
    value = cfg
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _get_nested(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = list(directory.glob(pattern))
    if not files:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        match = re.search(r"_step_(\d+)", path.stem)
        return (int(match.group(1)) if match else -1, path.stat().st_mtime)

    return max(files, key=sort_key)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open() as file:
        return json.load(file)


def _load_latest_csv_metrics(run_dir: Path) -> dict[str, Any]:
    csv_files = list(run_dir.glob("csv/**/metrics.csv"))
    if not csv_files:
        return {}
    metrics: dict[str, Any] = {}
    with max(csv_files, key=lambda path: path.stat().st_mtime).open(newline="") as file:
        for row in csv.DictReader(file):
            for key, value in row.items():
                if value not in ("", None):
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
    return metrics


def _resolved_name(cfg: DictConfig, path: str, default: str) -> str:
    try:
        value = _select(cfg, path)
        return str(value) if value else default
    except Exception:
        return default


def _branch_metadata(branch: str, split_cfg: DictConfig, default_quantizer: str) -> dict[str, Any]:
    quantizer = split_cfg.get(f"{branch}_quantizer") or default_quantizer
    if quantizer == "fsq":
        levels = list(split_cfg.get(f"fsq_{branch}_levels", []))
        return {
            "quantizer": quantizer,
            "levels": levels,
            "num_codes": int(np.prod(levels, dtype=np.int64)) if levels else 1,
            "active": bool(levels),
        }
    if quantizer == "vq":
        dim = int(split_cfg.get(f"vq_{branch}_dim", 0))
        return {
            "quantizer": quantizer,
            "dim": dim,
            "num_codes": int(split_cfg.get(f"vq_{branch}_num_codes", 1)),
            "active": dim > 0,
        }
    raise ValueError(f"Unsupported quantizer type {quantizer!r} for branch {branch!r}")


def _quantizer_metadata(cfg: DictConfig) -> dict[str, Any]:
    model_target = str(_select(cfg, "model._target_", ""))
    if model_target.endswith("FaissKMeansBaselineLightning"):
        return {
            "quantizer_mode": "kmeans",
            "quantizer_family": "kmeans",
            "total_codebook_size": int(_select(cfg, "model.num_codes")),
            "branches": {},
        }
    if model_target.endswith("DumbQuantizationBaselineLightning"):
        levels = list(_select(cfg, "model.q_levels", []))
        return {
            "quantizer_mode": "scalar_baseline",
            "quantizer_family": "scalar_baseline",
            "total_codebook_size": int(np.prod(levels, dtype=np.int64)),
            "branches": {},
        }

    model_kwargs = _select(cfg, "model.model_kwargs")
    if not _get_nested(model_kwargs, "quantization_enabled", True):
        return {
            "quantizer_mode": "continuous",
            "quantizer_family": "continuous",
            "total_codebook_size": None,
            "branches": {},
        }
    split_cfg = _get_nested(model_kwargs, "split_quantizer_cfg")
    if not split_cfg or split_cfg.get("mode", "vq") == "vq":
        num_codes = int(_get_nested(_get_nested(model_kwargs, "vq_kwargs"), "num_codes"))
        return {
            "quantizer_mode": "single_vq",
            "quantizer_family": "vq",
            "total_codebook_size": num_codes,
            "branches": {},
        }

    default_quantizer = split_cfg.get("quantizer", "fsq")
    branches = {
        branch: _branch_metadata(branch, split_cfg, default_quantizer)
        for branch in split_cfg.get("branch_order", ["mu", "alpha"])
    }
    active_branches = {name: metadata for name, metadata in branches.items() if metadata["active"]}
    families = {metadata["quantizer"] for metadata in active_branches.values()}
    return {
        "quantizer_mode": "split",
        "quantizer_family": next(iter(families)) if len(families) == 1 else "mixed",
        "total_codebook_size": int(
            np.prod([metadata["num_codes"] for metadata in active_branches.values()], dtype=np.int64)
        ),
        "branches": branches,
    }


def _histogram_features(histograms: dict[str, np.ndarray]) -> list[str]:
    suffix = "_orig_counts"
    return sorted(key[: -len(suffix)] for key in histograms if key.endswith(suffix))


def _histogram_residual_features(histograms: dict[str, np.ndarray]) -> list[str]:
    suffix = "_diff_counts"
    return sorted(
        key[: -len(suffix)]
        for key in histograms
        if key.endswith(suffix) and f"{key[: -len(suffix)]}_diff_bins" in histograms
    )


def _load_histograms(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path) as histograms:
        return {key: histograms[key] for key in histograms.files}


def _serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in record.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, (dict, list)):
            serialized[key] = json.dumps(value, sort_keys=True)
        else:
            serialized[key] = value
    return serialized


def _artifact_prefix(stage: str, group: str, suite: str | None = None) -> str:
    """Return the callback artifact stem for legacy or named test suites."""
    if suite is not None:
        return f"{stage}_{suite}_{group}_orbit"
    return f"{stage}_orbit" if group == "all" else f"{stage}_{group}_orbit"


def _artifact_dirs(run_dir: Path, artifact_dir: Path | None = None) -> list[Path]:
    """Return candidate directories containing saved ORBIT artifacts for a run."""
    if artifact_dir is not None:
        return [artifact_dir]
    candidates = [run_dir]
    best_ckpt = run_dir / "evaluation" / "best.ckpt"
    if best_ckpt.is_dir():
        candidates.append(best_ckpt)
    evaluation_dir = run_dir / "evaluation"
    if evaluation_dir.is_dir():
        candidates.extend(
            path
            for path in sorted(evaluation_dir.iterdir())
            if path.is_dir() and path != best_ckpt
        )
    return candidates


def _latest_artifact_file(
    run_dir: Path,
    subdir: str,
    pattern: str,
    artifact_dir: Path | None = None,
) -> Path | None:
    for candidate in _artifact_dirs(run_dir, artifact_dir):
        path = _latest_file(candidate / subdir, pattern)
        if path is not None:
            return path
    return None


def _collect_record(
    run_dir: Path,
    stage: str,
    group: str,
    suite: str | None = None,
    label_override: str | None = None,
    family_override: str | None = None,
    artifact_dir_override: Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    cfg = _load_config(run_dir / ".hydra" / "config.yaml")
    artifact_prefix = _artifact_prefix(stage, group, suite)
    histogram_path = _latest_artifact_file(
        run_dir,
        "saved_histograms",
        f"{artifact_prefix}_hists_step_*.npz",
        artifact_dir_override,
    )
    metrics_path = _latest_artifact_file(
        run_dir,
        "saved_metrics",
        f"{artifact_prefix}_metrics_step_*.json",
        artifact_dir_override,
    )
    metrics = _load_latest_csv_metrics(run_dir)
    metrics.update(_load_json(metrics_path))
    metadata = _quantizer_metadata(cfg)
    label = label_override or _resolved_name(
        cfg,
        "logger.wandb.name",
        _resolved_name(cfg, "task_name", run_dir.name),
    )
    record = {
        "job": run_dir.name,
        "label": label,
        "run_dir": run_dir,
        "histogram_path": histogram_path,
        "metrics_path": metrics_path,
        "artifact_dir": histogram_path.parents[1] if histogram_path else None,
        "artifact_dir_override": artifact_dir_override,
        "stage": stage,
        "group": group,
        "suite": suite,
        "plot_family": family_override or (
            "FAISS k-means" if metadata["quantizer_family"] == "kmeans" else None
        ),
        "seed": int(_cfg_get(cfg, "seed", 0)),
        "wandb_project": _select(cfg, "logger.wandb.project"),
        "wandb_group": _select(cfg, "logger.wandb.group"),
        **metadata,
        **metrics,
    }
    return record, _load_histograms(histogram_path)


def _run_dirs_from_multirun(multirun_dir: Path) -> list[Path]:
    config_paths = sorted(multirun_dir.glob("*/.hydra/config.yaml"))
    if not config_paths:
        raise SystemExit(f"No Hydra job directories found under {multirun_dir}")
    return [config_path.parent.parent for config_path in config_paths]


def _validate_run_dirs(run_dirs: list[Path]) -> list[Path]:
    if not run_dirs:
        raise SystemExit("No run directories were provided.")
    missing_configs = [
        run_dir for run_dir in run_dirs if not (run_dir / ".hydra" / "config.yaml").is_file()
    ]
    if missing_configs:
        formatted = "\n".join(f"  {path}" for path in missing_configs)
        raise SystemExit(
            "Every --run-dir must point at a single Hydra run output directory with "
            f".hydra/config.yaml. Missing configs:\n{formatted}"
        )
    return run_dirs


def _parse_named_runs(run_specs: list[list[str]]) -> list[tuple[Path, str | None]]:
    runs = []
    for spec in run_specs:
        if len(spec) not in (1, 2):
            raise SystemExit(
                "Each --run entry must be either '--run RUN_DIR' or "
                "'--run RUN_DIR LABEL'."
            )
        runs.append((Path(spec[0]), spec[1] if len(spec) == 2 else None))
    _validate_run_dirs([run_dir for run_dir, _ in runs])
    return runs


def _parse_family_runs(family_specs: list[list[str]]) -> list[tuple[Path, str | None, str]]:
    runs = []
    for spec in family_specs:
        if len(spec) < 2:
            raise SystemExit(
                "Each --family entry must be '--family FAMILY RUN_DIR [RUN_DIR ...]'."
            )
        family = spec[0]
        for run_dir in spec[1:]:
            runs.append((Path(run_dir), None, family))
    _validate_run_dirs([run_dir for run_dir, _, _ in runs])
    return runs


def _parse_artifact_family_runs(
    family_specs: list[list[str]],
) -> list[tuple[Path, str | None, str, Path]]:
    """Parse families whose runs use explicitly selected evaluation directories."""
    runs = []
    for spec in family_specs:
        if len(spec) < 3 or len(spec[1:]) % 2:
            raise SystemExit(
                "Each --artifact-family entry must be '--artifact-family FAMILY "
                "RUN_DIR ARTIFACT_DIR [RUN_DIR ARTIFACT_DIR ...]'."
            )
        family = spec[0]
        for index in range(1, len(spec), 2):
            runs.append((Path(spec[index]), None, family, Path(spec[index + 1])))
    _validate_run_dirs([run_dir for run_dir, _, _, _ in runs])
    missing_artifact_dirs = [
        artifact_dir for _, _, _, artifact_dir in runs if not artifact_dir.is_dir()
    ]
    if missing_artifact_dirs:
        formatted = "\n".join(f"  {path}" for path in missing_artifact_dirs)
        raise SystemExit(f"Explicit artifact directories do not exist:\n{formatted}")
    return runs


def _write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_records = [_serializable_record(record) for record in records]
    with (output_dir / "manifest.json").open("w") as file:
        json.dump(serializable_records, file, indent=2, sort_keys=True)
    fields = sorted({key for record in serializable_records for key in record})
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable_records)


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _add_multirun_metric_aliases(records: list[dict[str, Any]]) -> None:
    """Add stable plotting aliases across old and new ORBIT metric names."""
    alias_specs = {
        "plot_metrics/reco_mse_total": (
            "metrics/mse_total",
            "metrics/transformed_mse_total",
            "loss_reco_l2_per_value",
        ),
        "plot_metrics/transformed_mse_eta_scaled": (
            "metrics/transformed_mse_eta_scaled",
            "metrics/mse_L1T_PUPPIPart_Eta",
        ),
        "plot_metrics/transformed_mse_cos_phi": (
            "metrics/transformed_mse_cos_phi",
            "metrics/mse_L1T_PUPPIPart_Phi_cos",
        ),
        "plot_metrics/transformed_mse_sin_phi": (
            "metrics/transformed_mse_sin_phi",
            "metrics/mse_L1T_PUPPIPart_Phi_sin",
        ),
        "plot_metrics/transformed_mse_log_pt_shifted": (
            "metrics/transformed_mse_log_pt_shifted",
            "metrics/mse_L1T_PUPPIPart_PT",
        ),
        "plot_metrics/physical_mse_eta": (
            "metrics/mse_Eta",
            "metrics/physical_mse_Eta",
            "metrics/mse_eta",
        ),
        "plot_metrics/physical_mse_phi": (
            "metrics/mse_Phi",
            "metrics/physical_mse_Phi",
            "metrics/mse_phi",
        ),
        "plot_metrics/physical_mse_pT": (
            "metrics/mse_pT",
            "metrics/physical_mse_pT",
        ),
        "plot_metrics/physical_mse_total": (
            "metrics/physical_mse_total",
        ),
    }
    for record in records:
        for alias, keys in alias_specs.items():
            value = _first_present(record, keys)
            if value is not None:
                record[alias] = value


def _records_with_compression_ratio(
    records: list[dict[str, Any]], reference_bits: float
) -> list[dict[str, Any]]:
    """Return plotting records with marginal particle rate normalized to a payload size."""
    normalized_records = []
    for record in records:
        normalized = dict(record)
        rate = record.get("metrics/rate/marginal_bits_per_input_particle")
        if rate is not None:
            normalized["plot_metrics/compression_ratio"] = float(rate) / reference_bits
        normalized_records.append(normalized)
    return normalized_records


def _compression_ratio_figure(
    records: list[dict[str, Any]],
    *,
    reference_bits: float = 40.0,
    continuous_autoencoder_mse: float | None = None,
    show_title: bool = True,
):
    """Plot reconstruction error against the fraction of a fixed-width payload."""
    figure = plot_multirun_metric(
        _records_with_compression_ratio(records, reference_bits),
        "plot_metrics/reco_mse_total",
        "Reconstruction MSE",
        "Compression ratio vs. reconstruction MSE",
        log_x=False,
        log_y=True,
        x_metric="plot_metrics/compression_ratio",
        xlabel=f"Compressed rate / {reference_bits:g}-bit input payload",
        horizontal_reference=continuous_autoencoder_mse,
        horizontal_reference_label="Continuous autoencoder reference",
        show_title=show_title,
        scale_marker_by_codebook_size=True,
        marker_area_scale=PRESENTATION_CODEBOOK_MARKER_AREA_SCALE,
    )
    if figure is not None:
        # This is the longest x-axis label in the presentation collection.  A
        # wider canvas prevents its leading glyph from touching the tight
        # output bounding box for samples with narrow y-axis tick labels.
        figure.set_size_inches(9.5, 6.0, forward=True)
        figure.tight_layout()
    return figure


def _save_figures(
    records: list[dict[str, Any]],
    histogram_runs: list[dict],
    output_dir: Path,
    *,
    original_bits_per_input_particle: float = 43.0,
    compression_reference_bits: float = 40.0,
    continuous_autoencoder_mse: float | None = None,
    show_titles: bool = True,
    histogram_features: list[str] | None = None,
) -> None:
    figures = {
        "codebook_size_vs_reco_mse": plot_multirun_metric(
            records,
            "plot_metrics/reco_mse_total",
            "Reconstruction MSE",
            "Codebook size vs. reconstruction MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_mse_total": plot_multirun_metric(
            records,
            "metrics/mse_total",
            "Total reconstruction MSE",
            "Codebook size vs. reconstruction MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_utilization_total": plot_multirun_metric(
            records,
            "metrics/utilization_total",
            "Codebook utilization",
            "Codebook size vs. utilization",
            show_title=show_titles,
        ),
        "codebook_size_vs_marginal_entropy_bits_per_token": plot_multirun_metric(
            records,
            "metrics/entropy/marginal_bits_per_token",
            "Marginal entropy [bits/token]",
            "Codebook size vs. marginal token entropy",
            reference_fn=np.log2,
            reference_label=r"$\log_2 |\mathcal{C}|$ upper bound",
            show_title=show_titles,
        ),
        "codebook_size_vs_normalized_entropy": plot_multirun_metric(
            records,
            "metrics/entropy/normalized_to_log2_codebook",
            r"Marginal entropy / $\log_2 |\mathcal{C}|$",
            "Codebook size vs. normalized marginal entropy",
            show_title=show_titles,
        ),
        "codebook_size_vs_marginal_bits_per_event": plot_multirun_metric(
            records,
            "metrics/rate/marginal_bits_per_event",
            "Marginal rate [bits/event]",
            "Codebook size vs. marginal event rate",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_marginal_bits_per_input_particle": plot_multirun_metric(
            records,
            "metrics/rate/marginal_bits_per_input_particle",
            "Marginal rate [bits/input particle]",
            "Codebook size vs. marginal rate per input particle",
            log_y=True,
            show_title=show_titles,
        ),
        "marginal_bits_per_input_particle_vs_reco_mse": plot_multirun_metric(
            records,
            "plot_metrics/reco_mse_total",
            "Reconstruction MSE",
            "Marginal rate per input particle vs. reconstruction MSE",
            log_y=True,
            x_metric="metrics/rate/marginal_bits_per_input_particle",
            xlabel="Marginal rate [bits/input particle]",
            vertical_reference=original_bits_per_input_particle,
            vertical_reference_label=(
                f"Original particle representation ({original_bits_per_input_particle:g} bits)"
            ),
            horizontal_reference=continuous_autoencoder_mse,
            horizontal_reference_label="Continuous autoencoder reference",
            show_title=show_titles,
        ),
        "compression_ratio_40bit_vs_reco_mse": _compression_ratio_figure(
            records,
            reference_bits=compression_reference_bits,
            continuous_autoencoder_mse=continuous_autoencoder_mse,
            show_title=show_titles,
        ),
        "marginal_bits_per_event_vs_reco_mse": plot_multirun_metric(
            records,
            "plot_metrics/reco_mse_total",
            "Reconstruction MSE",
            "Marginal event rate vs. reconstruction MSE",
            log_y=True,
            x_metric="metrics/rate/marginal_bits_per_event",
            xlabel="Marginal rate [bits/event]",
            show_title=show_titles,
        ),
        "codebook_size_vs_val_loss": plot_multirun_metric(
            records,
            "val_loss_epoch",
            "Validation loss",
            "Codebook size vs. validation loss",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_transformed_mse_eta_scaled": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_eta_scaled",
            r"MSE $\eta/3$",
            r"Codebook size vs. transformed $\eta/3$ MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_transformed_mse_cos_phi": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_cos_phi",
            r"MSE $\cos\phi$",
            r"Codebook size vs. transformed $\cos\phi$ MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_transformed_mse_sin_phi": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_sin_phi",
            r"MSE $\sin\phi$",
            r"Codebook size vs. transformed $\sin\phi$ MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_transformed_mse_log_pt_shifted": plot_multirun_metric(
            records,
            "plot_metrics/transformed_mse_log_pt_shifted",
            r"MSE $\log(p_T)-1.8$",
            r"Codebook size vs. transformed $\log(p_T)-1.8$ MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_physical_mse_eta": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_eta",
            r"Physical MSE $\eta$",
            r"Codebook size vs. physical $\eta$ MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_physical_mse_phi": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_phi",
            r"Physical MSE $\phi$",
            r"Codebook size vs. physical $\phi$ MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_physical_mse_pT": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_pT",
            r"Physical MSE $p_T$",
            r"Codebook size vs. physical $p_T$ MSE",
            log_y=True,
            show_title=show_titles,
        ),
        "codebook_size_vs_physical_mse_total": plot_multirun_metric(
            records,
            "plot_metrics/physical_mse_total",
            "Mean physical MSE",
            "Codebook size vs. mean physical MSE",
            log_y=True,
            show_title=show_titles,
        ),
    }
    if histogram_runs:
        common_features = set(_histogram_features(histogram_runs[0]["histograms"]))
        for run in histogram_runs[1:]:
            common_features &= set(_histogram_features(run["histograms"]))
        feature_names = (
            [name for name in histogram_features if name in common_features]
            if histogram_features
            else sorted(common_features)
        )
        if feature_names:
            figures["combined_reconstruction_features"] = plot_multirun_feature_histograms(
                histogram_runs,
                feature_names,
                figsize_per_axis=(7.0, 5.2),
                show_titles=show_titles,
            )
        common_residual_features = set(_histogram_residual_features(histogram_runs[0]["histograms"]))
        for run in histogram_runs[1:]:
            common_residual_features &= set(_histogram_residual_features(run["histograms"]))
        residual_feature_names = (
            [name for name in histogram_features if name in common_residual_features]
            if histogram_features
            else sorted(common_residual_features)
        )
        if residual_feature_names:
            figures["combined_reconstruction_residuals"] = plot_multirun_residual_histograms(
                histogram_runs,
                residual_feature_names,
                show_titles=show_titles,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        if figure is None:
            continue
        figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def _first_nonempty(records: list[dict[str, Any]], key: str) -> str | None:
    for record in records:
        value = record.get(key)
        if value:
            return str(value)
    return None


def _wandb_upload_settings(args, records: list[dict[str, Any]], output_dir: Path) -> dict[str, str | None] | None:
    """Return a dedicated comparison-run identity when W&B credentials are available."""
    if args.no_wandb or not os.environ.get("WANDB_API_KEY"):
        return None
    return {
        "project": args.wandb_project
        or os.environ.get("WANDB_PROJECT")
        or _first_nonempty(records, "wandb_project")
        or "orbit-tokenizer",
        "name": args.wandb_name or f"orbit-multirun-{output_dir.name}",
        "group": args.wandb_group or os.environ.get("WANDB_GROUP") or "orbit-multirun-comparisons",
        "entity": args.wandb_entity or os.environ.get("WANDB_ENTITY"),
    }


def _upload_to_wandb(
    output_dir: Path,
    project: str,
    name: str | None,
    group: str | None,
    entity: str | None,
    records: list[dict[str, Any]],
    stage: str,
    comparison_group: str,
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("W&B upload requested, but wandb is not installed.") from exc

    run = wandb.init(
        project=project,
        name=name,
        group=group,
        entity=entity,
        job_type="multirun-comparison",
        config={
            "stage": stage,
            "artifact_group": comparison_group,
            "compared_run_count": len(records),
            "compared_runs": [str(record["run_dir"]) for record in records],
        },
    )
    try:
        payload = {}
        for image_path in sorted(output_dir.glob("*.png")):
            payload[f"multirun_plots/{image_path.stem}"] = wandb.Image(str(image_path))
        for artifact_path in (output_dir / "manifest.json", output_dir / "summary.csv"):
            if artifact_path.exists():
                run.save(str(artifact_path), base_path=str(output_dir))
        artifact = wandb.Artifact(
            name=f"orbit-multirun-{output_dir.name}-{run.id}",
            type="orbit-multirun-comparison",
            metadata={"stage": stage, "group": comparison_group, "run_count": len(records)},
        )
        for path in sorted(output_dir.glob("*.png")) + [
            output_dir / "manifest.json",
            output_dir / "summary.csv",
        ]:
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        if payload:
            run.log(payload)
    finally:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--multirun-dir",
        type=Path,
        help="Hydra multirun directory whose immediate children are run output dirs.",
    )
    source.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        action="append",
        help="One or more explicit Hydra run output directories to compare.",
    )
    source.add_argument(
        "--run",
        nargs="+",
        action="append",
        metavar="RUN_SPEC",
        help=(
            "Repeatable explicit run entry with optional label: "
            "--run RUN_DIR or --run RUN_DIR LABEL."
        ),
    )
    source.add_argument(
        "--family",
        nargs="+",
        action="append",
        metavar="FAMILY_SPEC",
        help=(
            "Repeatable family entry for grouped scalar plots: "
            "--family FAMILY RUN_DIR [RUN_DIR ...]."
        ),
    )
    source.add_argument(
        "--artifact-family",
        nargs="+",
        action="append",
        metavar="ARTIFACT_FAMILY_SPEC",
        help=(
            "Repeatable family with an explicit artifact directory per run: "
            "--artifact-family FAMILY RUN_DIR ARTIFACT_DIR "
            "[RUN_DIR ARTIFACT_DIR ...]."
        ),
    )
    parser.add_argument("--stage", choices=("val", "test"), default="val")
    parser.add_argument(
        "--suite",
        help="Named test suite to compare, e.g. training_like or tt_vs_gghbb.",
    )
    parser.add_argument("--group", default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-wandb", action="store_true", help="Keep comparison outputs local.")
    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Generate scalar comparison curves without loading combined histogram payloads.",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Suppress axes and panel titles for presentation-ready figures.",
    )
    parser.add_argument(
        "--histogram-feature",
        action="append",
        help=(
            "Restrict combined histogram and residual figures to this saved feature. "
            "Repeat to select and order multiple features."
        ),
    )
    parser.add_argument("--wandb-project", help="Override the dedicated W&B comparison project.")
    parser.add_argument("--wandb-name", help="Override the W&B comparison run name.")
    parser.add_argument("--wandb-group", help="Override the W&B comparison run group.")
    parser.add_argument("--wandb-entity", help="Override the W&B entity/team.")
    parser.add_argument(
        "--original-bits-per-input-particle",
        type=float,
        default=43.0,
        help="Vertical source-size reference in the particle rate-distortion plot (default: 43).",
    )
    parser.add_argument(
        "--compression-reference-bits",
        type=float,
        default=40.0,
        help=(
            "Denominator used for the normalized compression-ratio plot "
            "(default: 40 bits)."
        ),
    )
    parser.add_argument(
        "--continuous-autoencoder-mse",
        type=float,
        help="Optional horizontal MSE reference from a continuous autoencoder.",
    )
    args = parser.parse_args()

    if args.original_bits_per_input_particle <= 0:
        parser.error("--original-bits-per-input-particle must be positive")
    if args.compression_reference_bits <= 0:
        parser.error("--compression-reference-bits must be positive")
    if args.continuous_autoencoder_mse is not None and args.continuous_autoencoder_mse <= 0:
        parser.error("--continuous-autoencoder-mse must be positive")
    if args.suite is not None and args.stage != "test":
        parser.error("--suite can only be used with --stage test")

    if args.multirun_dir is not None:
        run_specs = [
            (run_dir, None, None, None)
            for run_dir in _run_dirs_from_multirun(args.multirun_dir)
        ]
        default_output_dir = args.multirun_dir / "comparisons" / args.stage / args.group
    elif args.artifact_family is not None:
        run_specs = _parse_artifact_family_runs(args.artifact_family)
        default_output_dir = Path.cwd() / "orbit_run_comparison" / args.stage / args.group
    elif args.family is not None:
        run_specs = [(*spec, None) for spec in _parse_family_runs(args.family)]
        default_output_dir = Path.cwd() / "orbit_run_comparison" / args.stage / args.group
    elif args.run is not None:
        run_specs = [
            (run_dir, label, None, None)
            for run_dir, label in _parse_named_runs(args.run)
        ]
        default_output_dir = Path.cwd() / "orbit_run_comparison" / args.stage / args.group
    else:
        run_dirs = _validate_run_dirs([run_dir for group in args.run_dir for run_dir in group])
        run_specs = [(run_dir, None, None, None) for run_dir in run_dirs]
        default_output_dir = Path.cwd() / "orbit_run_comparison" / args.stage / args.group

    records = []
    loaded_histograms = []
    for run_dir, label_override, family_override, artifact_dir_override in run_specs:
        record, histograms = _collect_record(
            run_dir,
            args.stage,
            args.group,
            args.suite,
            label_override,
            family_override,
            artifact_dir_override,
        )
        records.append(record)
        loaded_histograms.append(None if args.no_histograms else histograms)

    label_counts = Counter(record["label"] for record in records)
    for record in records:
        if label_counts[record["label"]] > 1:
            record["label"] = f"{record['label']} [{record['job']}]"
    histogram_runs = [
        {
            "label": record["label"],
            "plot_color": multirun_color(record.get("plot_family") or record["label"]),
            "histograms": histograms,
        }
        for record, histograms in zip(records, loaded_histograms)
        if histograms is not None
    ]

    output_dir = args.output_dir or default_output_dir
    _add_multirun_metric_aliases(records)
    _write_outputs(records, output_dir)
    _save_figures(
        records,
        histogram_runs,
        output_dir,
        original_bits_per_input_particle=args.original_bits_per_input_particle,
        compression_reference_bits=args.compression_reference_bits,
        continuous_autoencoder_mse=args.continuous_autoencoder_mse,
        show_titles=not args.no_titles,
        histogram_features=args.histogram_feature,
    )
    wandb_settings = _wandb_upload_settings(args, records, output_dir)
    if wandb_settings is not None:
        _upload_to_wandb(
            output_dir,
            project=wandb_settings["project"],
            name=wandb_settings["name"],
            group=wandb_settings["group"],
            entity=wandb_settings["entity"],
            records=records,
            stage=args.stage,
            comparison_group=args.group,
        )
    elif not args.no_wandb:
        print("W&B credentials were not found; kept comparison outputs local.")
    print(f"Collected {len(records)} runs into {output_dir}")


if __name__ == "__main__":
    main()
