#!/usr/bin/env python
"""Build the title-free ORBIT writeup plot collection from cached evaluations.

The script deliberately follows the existing presentation-plot workflow: its
inputs are ordinary command-line arguments and generated manifests, while the
layout and style remain in Python where they can be iterated on quickly.
Inference is kept out of this renderer.  Expensive evaluators write caches once;
this script can then be rerun without loading a checkpoint or touching a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

from gabbro.plotting.orbit import (
    CODEBOOK_SIZE_COLORS,
    ORIGINAL_COLOR,
    RECONSTRUCTED_COLOR,
    RUN_COLORS,
    multirun_color,
    multirun_display_label,
    multirun_marker,
)


PID_CLASS_NAMES = (
    "neutral hadron",
    "photon",
    "negative hadron",
    "positive hadron",
    "electron",
    "positron",
    "muon",
    "antimuon",
)
PID_DISPLAY_LABELS = (
    r"$h^0$",
    r"$\gamma$",
    r"$h^-$",
    r"$h^+$",
    r"$e^-$",
    r"$e^+$",
    r"$\mu^-$",
    r"$\mu^+$",
)
VQ_STE_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
LEGEND_FONT_SIZE = 16
LEGEND_TITLE_FONT_SIZE = 17
# The golden overview and codebook-size sweep have substantially denser
# legends than the other writeup figures.  Keep their labels clear of the
# plotted curves without shrinking the legends used by the simpler panels.
GOLDEN_SUMMARY_LEGEND_FONT_SIZE = 11.5
DENSE_LEGEND_FONT_SIZE = 12.5
DENSE_LEGEND_TITLE_FONT_SIZE = 13.5
CAMPAIGN_MODELS = (
    {
        "experiment": "orbit_campaign_full500_tt_vq4096_5m",
        "label": "Full 500, ttbar",
        "quantizer": "VQ-STE",
        "codebook_size": 4096,
        "selection": "leading 500, no PUPPIW cut",
        "training_sample": "ttbar",
        "loss": "unweighted",
        "wandb_train_id": "68mo7hin",
        "wandb_sm_test_id": "xgw1x8tx",
    },
    {
        "experiment": "orbit_campaign_full500_sm_vq4096_5m",
        "label": "Full 500, SM",
        "quantizer": "VQ-STE",
        "codebook_size": 4096,
        "selection": "leading 500, no PUPPIW cut",
        "training_sample": "balanced SM mixture",
        "loss": "unweighted",
        "wandb_train_id": "kit9ge98",
        "wandb_sm_test_id": "yo9spf60",
    },
    {
        "experiment": "orbit_campaign_full500_sm_pidweighted_vq4096_5m",
        "label": "Full 500, SM, PID weighted",
        "quantizer": "VQ-STE",
        "codebook_size": 4096,
        "selection": "leading 500, no PUPPIW cut",
        "training_sample": "balanced SM mixture",
        "loss": "PID-frequency weighted",
        "wandb_train_id": "99riqi8k",
        "wandb_sm_test_id": "v9jr9zcb",
    },
    {
        "experiment": "orbit_campaign_puppi128_sm_vq4096_5m",
        "label": "PUPPIW 128, SM",
        "quantizer": "VQ-STE",
        "codebook_size": 4096,
        "selection": "PUPPIW > 0.05, leading 128",
        "training_sample": "balanced SM mixture",
        "loss": "unweighted",
        "wandb_train_id": "7ptnx3mu",
        "wandb_sm_test_id": "hn8j9ykw",
    },
    {
        "experiment": "orbit_campaign_puppi128_tt_vq4096_5m",
        "label": "PUPPIW 128, ttbar",
        "quantizer": "VQ-STE",
        "codebook_size": 4096,
        "selection": "PUPPIW > 0.05, leading 128",
        "training_sample": "ttbar",
        "loss": "unweighted",
        "wandb_train_id": "6hpvryxr",
        "wandb_sm_test_id": "63bzl5f5",
    },
    {
        "experiment": "orbit_campaign_puppi128_sm_pidweighted_vq4096_5m",
        "label": "PUPPIW 128, SM, PID weighted",
        "quantizer": "VQ-STE",
        "codebook_size": 4096,
        "selection": "PUPPIW > 0.05, leading 128",
        "training_sample": "balanced SM mixture",
        "loss": "PID-frequency weighted",
        "wandb_train_id": "5cp8rdlm",
        "wandb_sm_test_id": "i10kxwl9",
    },
    {
        "experiment": "orbit_campaign_full500_sm_vq8192_5m",
        "label": "Full 500, SM, 8192 codes",
        "quantizer": "VQ-STE",
        "codebook_size": 8192,
        "selection": "leading 500, no PUPPIW cut",
        "training_sample": "balanced SM mixture",
        "loss": "unweighted",
        "wandb_train_id": None,
        "wandb_sm_test_id": None,
        "artifact_slug": "orbit-campaign-5m-full500-sm-vq8192",
    },
    {
        "experiment": "orbit_campaign_full500_sm_faiss4096_5m",
        "label": "Full 500, balanced FAISS fit",
        "quantizer": "FAISS k-means",
        "codebook_size": 4096,
        "selection": "leading 500, no PUPPIW cut",
        "training_sample": "10M-particle balanced fit (2M each QCD/ttbar/V+jets/VV/DY)",
        "loss": "not applicable",
        "wandb_train_id": None,
        "wandb_sm_test_id": None,
        "artifact_slug": "orbit-campaign-5m-full500-sm-faiss4096",
    },
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-manifest", type=Path, required=True)
    parser.add_argument("--campaign-summary-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--campaign-metric",
        action="append",
        default=[],
        metavar="EXPERIMENT=VALUE",
        help="Explicitly supply one SM-test loss_reco_l2_per_value value.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--golden-pid-confusion", type=Path)
    parser.add_argument("--golden-pid-residuals", type=Path)
    parser.add_argument("--downstream-run-dir", type=Path)
    parser.add_argument("--higgs-cache", type=Path)
    parser.add_argument("--tau32-cache", type=Path)
    parser.add_argument(
        "--met-cache",
        type=Path,
        help="Saved orbit histogram NPZ for the SM-mixture MET comparison.",
    )
    parser.add_argument(
        "--met-run-dir",
        type=Path,
        help="Run directory for the SM-mixture VQ-STE model used by the MET cache.",
    )
    parser.add_argument("--jet-residual-cache", type=Path)
    parser.add_argument("--sm-pid-residuals", type=Path)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        help="Optional directory containing the 5M-versus-200k comparison products.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty record list in {path}")
    return records


def save_figure(figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 300})):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def add_writeup_header(
    figure,
    top_axes,
    *,
    training_text: str,
    test_text: str,
    selection_text: str,
) -> None:
    """Add writeup metadata without allowing labels to stagger the panels.

    Axis-attached experiment labels participate in constrained layout.  On a
    multi-column figure this can move only the labelled axes downward.  Reserve
    one figure-level band instead, then anchor all header text to the already
    aligned top-row axes.
    """
    axes = list(top_axes)
    if not axes:
        raise ValueError("top_axes must contain at least one axis")
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.91))
    figure.canvas.draw()

    last = axes[-1].get_position()
    figure.text(
        last.x1,
        0.985,
        f"Train: {training_text}  |  Test: {test_text}",
        ha="right",
        va="top",
        fontsize=13,
    )
    figure.text(
        last.x1,
        0.958,
        selection_text,
        ha="right",
        va="top",
        fontsize=13,
    )


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _family_series(records: list[dict[str, Any]], x_key: str, y_key: str):
    grouped = defaultdict(list)
    for record in records:
        x, y = _finite(record.get(x_key)), _finite(record.get(y_key))
        if x is not None and y is not None:
            grouped[str(record.get("plot_family") or record.get("label"))].append((x, y))
    return grouped


def missing_golden_metric_records(records: list[dict[str, Any]]) -> list[str]:
    """Return scientific descriptions that cannot appear in every summary panel."""
    required = (
        "total_codebook_size",
        "plot_metrics/reco_mse_total",
        "metrics/utilization_total",
        "metrics/rate/marginal_bits_per_input_particle",
    )
    missing = []
    for record in records:
        if not any(_finite(record.get(key)) is None for key in required):
            continue
        family = str(record.get("plot_family") or "unknown quantizer")
        size = record.get("total_codebook_size")
        missing.append(f"{family}, {size} codes" if size is not None else family)
    return missing


def plot_golden_summary(records: list[dict[str, Any]]):
    """Return the requested one-row MSE/utilization/rate-distortion figure."""
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.6), constrained_layout=True)
    specs = (
        (
            "total_codebook_size",
            "plot_metrics/reco_mse_total",
            "Nominal codebook size",
            "Reconstruction MSE",
            True,
        ),
        (
            "total_codebook_size",
            "metrics/utilization_total",
            "Nominal codebook size",
            "Active-code utilization",
            False,
        ),
        (
            "metrics/rate/marginal_bits_per_input_particle",
            "plot_metrics/reco_mse_total",
            "Marginal rate / 40-bit input payload",
            "Reconstruction MSE",
            True,
        ),
    )
    for axis, (x_key, y_key, xlabel, ylabel, log_y) in zip(axes, specs, strict=True):
        plot_records = records
        if x_key.startswith("metrics/rate"):
            plot_records = [
                {**row, x_key: _finite(row.get(x_key)) / 40.0}
                for row in records
                if _finite(row.get(x_key)) is not None
            ]
        for family, points in sorted(_family_series(plot_records, x_key, y_key).items()):
            points = sorted(points)
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=multirun_color(family),
                marker=multirun_marker(family),
                markersize=7.5,
                linewidth=1.5,
                label=multirun_display_label(family),
            )
        axis.set(xlabel=xlabel, ylabel=ylabel)
        if x_key == "total_codebook_size":
            axis.set_xscale("log", base=2)
        if log_y:
            axis.set_yscale("log")
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        labels,
        loc="upper right",
        ncol=1,
        fontsize=GOLDEN_SUMMARY_LEGEND_FONT_SIZE,
        frameon=False,
        markerscale=1.35,
        handlelength=1.6,
        labelspacing=0.3,
    )
    return figure


def select_vq_ste_size_records(
    records: list[dict[str, Any]], *, require_complete: bool = True
) -> list[dict[str, Any]]:
    by_size = {}
    for record in records:
        if record.get("plot_family") != "VQ STE":
            continue
        size = int(record["total_codebook_size"])
        if size in VQ_STE_SIZES and record.get("histogram_path"):
            by_size[size] = record
    missing = sorted(set(VQ_STE_SIZES) - set(by_size))
    if missing and require_complete:
        raise ValueError(f"Missing VQ-STE histogram caches for codebook sizes: {missing}")
    return [by_size[size] for size in VQ_STE_SIZES if size in by_size]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: np.asarray(source[key]).copy() for key in source.files}


def _stairs(axis, counts, edges, **kwargs):
    return axis.stairs(np.asarray(counts, dtype=float), np.asarray(edges), **kwargs)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def plot_vq_size_feature_sweep(
    records: list[dict[str, Any]], *, require_complete: bool = True
):
    """Plot physical features and residual evolution across VQ-STE sizes."""
    selected = select_vq_ste_size_records(records, require_complete=require_complete)
    payloads = [(int(row["total_codebook_size"]), _load_npz(Path(row["histogram_path"]))) for row in selected]
    features = (
        ("physical_pT", r"$p_{\mathrm{T}}$ [GeV]", r"$p_T^\mathrm{reco}-p_T^\mathrm{orig}$ [GeV]", True),
        ("physical_Eta", r"$\eta$", r"$\eta^\mathrm{reco}-\eta^\mathrm{orig}$", False),
        ("physical_Phi", r"$\phi$", r"wrapped $\phi^\mathrm{reco}-\phi^\mathrm{orig}$", False),
    )
    hep.style.use("CMS")
    figure = plt.figure(figsize=(17.0, 10.0), constrained_layout=True)
    outer = figure.add_gridspec(2, 3, height_ratios=(1.45, 0.9))
    for column_index, (key, xlabel, residual_label, log_x) in enumerate(features):
        top = outer[0, column_index].subgridspec(
            2, 1, height_ratios=(3.2, 1.0), hspace=0.04
        )
        hist_axis = figure.add_subplot(top[0])
        ratio_axis = figure.add_subplot(top[1], sharex=hist_axis)
        residual_axis = figure.add_subplot(outer[1, column_index])
        reference = payloads[0][1]
        _stairs(
            hist_axis,
            reference[f"{key}_orig_counts"],
            reference[f"{key}_bins"],
            fill=True,
            alpha=0.28,
            color=ORIGINAL_COLOR,
            label="Original",
        )
        for size, payload in payloads:
            color = CODEBOOK_SIZE_COLORS[str(size)]
            _stairs(
                hist_axis,
                payload[f"{key}_reco_counts"],
                payload[f"{key}_bins"],
                color=color,
                linewidth=1.5,
                label=f"{size}",
            )
            ratio = _safe_ratio(payload[f"{key}_reco_counts"], payload[f"{key}_orig_counts"])
            _stairs(ratio_axis, ratio, payload[f"{key}_bins"], color=color, linewidth=1.25)
            _stairs(
                residual_axis,
                payload[f"{key}_diff_counts"],
                payload[f"{key}_diff_bins"],
                color=color,
                linewidth=1.5,
                label=f"{size}",
            )
        hist_axis.set_ylabel("Particles" if column_index == 0 else "")
        hist_axis.set_yscale("log")
        hist_axis.tick_params(labelbottom=False)
        ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ratio_axis.set(
            xlabel=xlabel,
            ylabel="Dec./orig." if column_index == 0 else "",
            ylim=(0.5, 1.5),
        )
        residual_axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        residual_axis.set(
            xlabel=residual_label,
            ylabel="Particles" if column_index == 0 else "",
        )
        residual_axis.set_yscale("log")
        if log_x:
            hist_axis.set_xscale("log")
            ratio_axis.set_xscale("log")
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.axes[0].legend(
        handles,
        labels,
        title="VQ-STE codes",
        loc="upper right",
        ncol=2,
        fontsize=DENSE_LEGEND_FONT_SIZE,
        title_fontsize=DENSE_LEGEND_TITLE_FONT_SIZE,
        frameon=False,
    )
    return figure


def plot_pid_confusion(path: Path):
    if path.suffix == ".npz":
        payload = _load_npz(path)
        if "counts" in payload:
            counts = np.asarray(payload["counts"], dtype=np.int64)
        else:
            original_pid = np.asarray(payload["original_pid"], dtype=int)
            decoded_pid = np.asarray(payload["decoded_pid"], dtype=int)
            counts = np.zeros(
                (len(PID_CLASS_NAMES), len(PID_CLASS_NAMES)), dtype=np.int64
            )
            np.add.at(counts, (original_pid, decoded_pid), 1)
        labels = list(PID_DISPLAY_LABELS)
    else:
        payload = json.loads(path.read_text())
        counts = np.asarray(payload["counts"], dtype=np.int64)
        raw_labels = payload.get("truth_labels", PID_CLASS_NAMES)
        labels = (
            list(PID_DISPLAY_LABELS)
            if len(raw_labels) == len(PID_DISPLAY_LABELS)
            else [str(label).replace("_", " ") for label in raw_labels]
        )
    normalized = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts, dtype=float),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )
    hep.style.use("CMS")
    figure, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = normalized[row, column]
            if value >= 0.001 or row == column:
                axis.text(
                    column,
                    row,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.55 else "black",
                )
    axis.set(
        xlabel="Decoded PID",
        ylabel="Original PID",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=0, labelsize=15)
    axis.tick_params(axis="y", labelsize=15)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Row-normalized fraction", fontsize=17)
    colorbar.ax.tick_params(labelsize=14)
    return figure, counts, normalized


def plot_pid_residuals(path: Path):
    payload = _load_npz(path)
    original_pid = payload["original_pid"].astype(int)
    specs = (
        ("delta_eta", r"$\eta^\mathrm{reco}-\eta^\mathrm{orig}$", (-2.0, 2.0)),
        ("delta_phi", r"wrapped $\phi^\mathrm{reco}-\phi^\mathrm{orig}$", (-1.0, 1.0)),
        ("delta_log_pt", r"$\log(p_T^\mathrm{reco}/p_T^\mathrm{orig})$", (-1.5, 1.5)),
    )
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.2), constrained_layout=True)
    for axis_index, (axis, (key, label, value_range)) in enumerate(
        zip(axes, specs, strict=True)
    ):
        edges = np.linspace(*value_range, 81)
        for pid, class_name in enumerate(PID_DISPLAY_LABELS):
            values = payload[key][original_pid == pid]
            values = values[np.isfinite(values)]
            if len(values):
                axis.hist(
                    values,
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=RUN_COLORS[pid],
                    label=class_name,
                )
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(
            xlabel=label,
            ylabel="Normalized density" if axis_index == 0 else "",
            xlim=value_range,
        )
    axes[-1].legend(fontsize=LEGEND_FONT_SIZE, frameon=False, ncol=2)
    return figure


def _parse_explicit_metrics(values: list[str]) -> dict[str, float]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected EXPERIMENT=VALUE, got {value!r}")
        experiment, scalar = value.split("=", 1)
        parsed[experiment] = float(scalar)
    return parsed


def _summary_metric(path: Path, *, require_sm_suite: bool = False) -> float | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if require_sm_suite and not any(
        key.startswith("test_metrics/sm_mixture/") for key in payload
    ):
        return None
    return _finite(payload.get("test_metrics/loss_reco_l2_per_value"))


def discover_campaign_metrics(roots: list[Path], explicit: dict[str, float]) -> dict[str, float]:
    result = dict(explicit)
    summaries = []
    for root in roots:
        summaries.extend(root.glob("**/wandb-summary.json"))
    for model in CAMPAIGN_MODELS:
        experiment = model["experiment"]
        if experiment in result:
            continue
        path_tokens = (experiment, experiment.replace("_", "-"), model.get("artifact_slug"))
        candidates = [
            path
            for path in summaries
            if any(token and token in str(path) for token in path_tokens)
        ]
        for path in sorted(set(candidates), key=lambda item: item.stat().st_mtime, reverse=True):
            metric = _summary_metric(
                path,
                require_sm_suite="sm_mixture" not in str(path),
            )
            if metric is not None:
                result[experiment] = metric
                break
    return result


def write_ablation_table(metrics: dict[str, float], output_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {
            **{key: value for key, value in model.items() if key != "artifact_slug"},
            "test_sample": "balanced SM mixture",
            "reconstruction_mse": metrics.get(model["experiment"]),
        }
        for model in CAMPAIGN_MODELS
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with (output_dir / "particle_reconstruction_ablation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "particle_reconstruction_ablation.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        r"\begin{tabular}{llllr}",
        r"\toprule",
        r"Representation & Fit/train sample & Quantizer & Loss & Test MSE \\",
        r"\midrule",
    ]
    for row in rows:
        mse = "--" if row["reconstruction_mse"] is None else f"{row['reconstruction_mse']:.6g}"
        values = [row["selection"], row["training_sample"], f"{row['quantizer']} {row['codebook_size']}", row["loss"], mse]
        escaped = [str(value).replace("_", r"\_").replace(">", r"$>$") for value in values]
        lines.append(" & ".join(escaped) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output_dir / "particle_reconstruction_ablation.tex").write_text("\n".join(lines) + "\n")
    return rows


def _hist_ratio_panel(
    axis,
    ratio_axis,
    original,
    decoded,
    bins,
    xlabel,
    *,
    log_y=False,
    show_y_labels=True,
):
    original = np.asarray(original, dtype=float)
    decoded = np.asarray(decoded, dtype=float)
    finite_original = original[np.isfinite(original)]
    finite_decoded = decoded[np.isfinite(decoded)]
    original_counts, edges = np.histogram(finite_original, bins=bins)
    decoded_counts, _ = np.histogram(finite_decoded, bins=edges)
    norm_original = original_counts / max(original_counts.sum(), 1)
    norm_decoded = decoded_counts / max(decoded_counts.sum(), 1)
    _stairs(axis, norm_original, edges, fill=True, alpha=0.35, color=ORIGINAL_COLOR, label="Original")
    _stairs(axis, norm_decoded, edges, color=RECONSTRUCTED_COLOR, linewidth=2.0, label="Decoded")
    _stairs(ratio_axis, _safe_ratio(norm_decoded, norm_original), edges, color=RECONSTRUCTED_COLOR, linewidth=1.5)
    axis.set_ylabel("Normalized events" if show_y_labels else "")
    if log_y:
        axis.set_yscale("log")
    ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ratio_axis.set(
        xlabel=xlabel,
        ylabel="Dec./orig." if show_y_labels else "",
        ylim=(0.5, 1.5),
    )
    return edges, original_counts, decoded_counts


def plot_higgs(cache: Path):
    rows = list(csv.DictReader(cache.open(newline="")))
    values = {key: np.asarray([float(row[key]) for row in rows]) for key in ("n_jets_original", "n_jets_decoded", "resolved_original", "resolved_decoded")}
    max_jets = int(max(values["n_jets_original"].max(initial=0), values["n_jets_decoded"].max(initial=0)))
    hep.style.use("CMS")
    figure = plt.figure(figsize=(13.4, 5.6), constrained_layout=True)
    outer = figure.add_gridspec(1, 2)
    saved = {}
    for index, (orig_key, reco_key, bins, label, log_y) in enumerate((
        ("n_jets_original", "n_jets_decoded", np.arange(-0.5, max_jets + 1.5), r"Number of AK4 jets ($p_T\geq30$ GeV)", True),
        ("resolved_original", "resolved_decoded", np.arange(40.0, 405.0, 5.0), r"Higgs candidate (dijet) mass [GeV]", False),
    )):
        grid = outer[index].subgridspec(2, 1, height_ratios=(3.2, 1), hspace=0.04)
        axis = figure.add_subplot(grid[0])
        ratio_axis = figure.add_subplot(grid[1], sharex=axis)
        edges, orig_counts, reco_counts = _hist_ratio_panel(
            axis,
            ratio_axis,
            values[orig_key],
            values[reco_key],
            bins,
            label,
            log_y=log_y,
            show_y_labels=index == 0,
        )
        axis.tick_params(labelbottom=False)
        saved[f"panel_{index}_bins"] = edges
        saved[f"panel_{index}_original_counts"] = orig_counts
        saved[f"panel_{index}_decoded_counts"] = reco_counts
    figure.axes[0].legend(fontsize=LEGEND_FONT_SIZE, frameon=False)
    return figure, values | saved


def plot_tau32(cache: Path):
    values = _load_npz(cache)
    original_counts = values["original_ak8_jet_count"]
    decoded_counts = values["decoded_ak8_jet_count"]
    max_jets = int(max(original_counts.max(initial=0), decoded_counts.max(initial=0)))
    hep.style.use("CMS")
    figure = plt.figure(figsize=(13.4, 5.6), constrained_layout=True)
    outer = figure.add_gridspec(1, 2)
    for index, (original, decoded, bins, label, log_y) in enumerate((
        (original_counts, decoded_counts, np.arange(-0.5, max_jets + 1.5), r"Number of AK8 jets ($p_T\geq450$ GeV)", True),
        (values["original_tau32"], values["decoded_tau32"], np.linspace(0, 1.2, 51), r"AK8 jet $\tau_{32}$", False),
    )):
        grid = outer[index].subgridspec(2, 1, height_ratios=(3.2, 1), hspace=0.04)
        axis = figure.add_subplot(grid[0])
        ratio_axis = figure.add_subplot(grid[1], sharex=axis)
        _hist_ratio_panel(
            axis,
            ratio_axis,
            original,
            decoded,
            bins,
            label,
            log_y=log_y,
            show_y_labels=index == 0,
        )
        axis.tick_params(labelbottom=False)
    figure.axes[0].legend(fontsize=LEGEND_FONT_SIZE, frameon=False)
    return figure, values


def plot_met(cache: Path):
    """Plot cached original and decoded missing-transverse-energy histograms."""
    values = _load_npz(cache)
    bins = values["physical_missing_et_bins"]
    original = values["physical_missing_et_orig_counts"]
    decoded = values["physical_missing_et_reco_counts"]
    residual_bins = values["physical_missing_et_diff_bins"]
    residual_counts = values["physical_missing_et_diff_counts"]
    ratio = _safe_ratio(decoded, original)

    hep.style.use("CMS")
    figure = plt.figure(figsize=(14.0, 6.5), constrained_layout=True)
    outer = figure.add_gridspec(1, 2)
    grid = outer[0].subgridspec(2, 1, height_ratios=(3.2, 1.0), hspace=0.04)
    axis = figure.add_subplot(grid[0])
    ratio_axis = figure.add_subplot(grid[1], sharex=axis)
    residual_axis = figure.add_subplot(outer[1])

    _stairs(
        axis,
        original,
        bins,
        fill=True,
        alpha=0.35,
        color=ORIGINAL_COLOR,
        label="Original",
    )
    _stairs(
        axis,
        decoded,
        bins,
        color=RECONSTRUCTED_COLOR,
        linewidth=2.0,
        label="Decoded",
    )
    _stairs(
        ratio_axis,
        ratio,
        bins,
        color=RECONSTRUCTED_COLOR,
        linewidth=1.5,
    )
    axis.set(ylabel="Normalized density", yscale="log")
    axis.tick_params(labelbottom=False)
    axis.legend(fontsize=LEGEND_FONT_SIZE, frameon=False)
    ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ratio_axis.set(
        xlabel=r"Missing transverse energy [GeV]",
        ylabel="Dec./orig.",
        xlim=(0.0, 600.0),
        ylim=(0.8, 1.2),
    )
    _stairs(
        residual_axis,
        residual_counts,
        residual_bins,
        color=RECONSTRUCTED_COLOR,
        linewidth=2.0,
    )
    residual_axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    residual_axis.set(
        xlabel="Decoded − original\nmissing transverse energy [GeV]",
        ylabel="Normalized density",
        xlim=(-200.0, 200.0),
        yscale="log",
    )
    return figure, {
        "bins": bins,
        "original_counts": original,
        "decoded_counts": decoded,
        "decoded_over_original": ratio,
        "residual_bins": residual_bins,
        "residual_counts": residual_counts,
    }


def plot_jet_residuals(cache: Path, prefix: str):
    values = _load_npz(cache)
    specs = (
        ("pt", r"$(p_T^\mathrm{dec.}-p_T^\mathrm{orig.})/p_T^\mathrm{orig.}$", (-1.0, 1.0)),
        ("eta", r"$\eta^\mathrm{dec.}-\eta^\mathrm{orig.}$", (-1.0, 1.0)),
        ("phi", r"wrapped $\phi^\mathrm{dec.}-\phi^\mathrm{orig.}$", (-1.0, 1.0)),
    )
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 5.1), constrained_layout=True)
    for axis_index, (axis, (feature, label, value_range)) in enumerate(
        zip(axes, specs, strict=True)
    ):
        residual = values[f"{prefix}_delta_{feature}"]
        residual = residual[np.isfinite(residual)]
        axis.hist(
            residual,
            bins=np.linspace(*value_range, 81),
            density=True,
            histtype="step",
            linewidth=2.0,
            color=RECONSTRUCTED_COLOR,
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(
            xlabel=label,
            ylabel="Normalized density" if axis_index == 0 else "",
            xlim=value_range,
        )
    return figure, values


def validate_downstream_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        config_path = run_dir / ".hydra/config.yaml"
    checkpoint = run_dir / "checkpoints/best.ckpt"
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Expected config and checkpoints/best.ckpt under {run_dir}")
    config = yaml.safe_load(config_path.read_text())
    max_length = int(config["data"]["max_sequence_length"])
    sequence_type = str(config["data"]["sequence_type"])
    num_codes = int(config["model"]["model_kwargs"]["vq_kwargs"]["num_codes"])
    task_name = str(config.get("task_name", ""))
    registered = next(
        (
            model
            for model in CAMPAIGN_MODELS
            if model["experiment"] == task_name
            and model["quantizer"] == "VQ-STE"
            and model["training_sample"] == "balanced SM mixture"
            and model["loss"] == "unweighted"
            and model["codebook_size"] == num_codes
        ),
        None,
    )
    if max_length != 500 or sequence_type != "particle_full" or registered is None:
        raise ValueError(
            "Downstream model must be a registered unweighted, full-500, "
            "SM-mixture-trained VQ-STE model"
        )
    return {
        **registered,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "task_name": task_name,
    }


def _provenance_row(figure_id: str, model: dict[str, Any], **overrides) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "experiment": model.get("experiment"),
        "run_or_wandb_id": model.get("wandb_train_id"),
        "checkpoint": model.get("checkpoint"),
        "quantizer": model.get("quantizer"),
        "codebook_size": model.get("codebook_size"),
        "particle_selection": model.get("selection"),
        "training_sample": model.get("training_sample"),
        "test_sample": overrides.pop("test_sample", None),
        "events": overrides.pop("events", None),
        "jet_definition": overrides.pop("jet_definition", None),
        "matching": overrides.pop("matching", None),
        **overrides,
    }


def write_model_index(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = tuple(rows[0])
    with (output_dir / "model_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "model_index.json").write_text(json.dumps(rows, indent=2) + "\n")
    md = ["# Model and sample index", "", "| Figure | Model | Quantizer | Codes | Train | Test | Selection |", "|---|---|---:|---:|---|---|---|"]
    for row in rows:
        md.append(f"| {row['figure_id']} | {row['experiment'] or row['run_or_wandb_id']} | {row['quantizer']} | {row['codebook_size']} | {row['training_sample']} | {row['test_sample']} | {row['particle_selection']} |")
    (output_dir / "model_index.md").write_text("\n".join(md) + "\n")
    tex = [
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        "Figure & Model & Quantizer & Codes & Train & Test & Selection \\\\",
        r"\midrule",
    ]
    for row in rows:
        values = [row["figure_id"], row["experiment"] or row["run_or_wandb_id"], row["quantizer"], row["codebook_size"], row["training_sample"], row["test_sample"], row["particle_selection"]]
        tex.append(
            " & ".join(str(value or "--").replace("_", r"\_") for value in values)
            + " \\\\"
        )
    tex.extend((r"\bottomrule", r"\end{tabular}"))
    (output_dir / "model_index.tex").write_text("\n".join(tex) + "\n")


def write_readme(output_dir: Path, products: list[dict[str, Any]], missing: list[str]) -> None:
    lines = [
        "# ORBIT writeup plots",
        "",
        "All figures are title-free and provided as PDF and 300-dpi PNG. Numerical caches and provenance sit beside the figures.",
        "",
        "## Products",
        "",
    ]
    for product in products:
        links = ", ".join(f"[{path.name}]({path.relative_to(output_dir)})" for path in product["paths"])
        lines.append(f"- **{product['label']}**: {links}")
    lines.extend(("", "## Model provenance", "", "See [model_index.md](model_index.md), [CSV](model_index.csv), [JSON](model_index.json), or [LaTeX](model_index.tex)."))
    if missing:
        lines.extend(("", "## Pending inputs", ""))
        lines.extend(f"- {item}" for item in missing)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.golden_manifest.resolve())
    if len(records) != 46:
        raise ValueError(f"Expected 46 golden records, found {len(records)}")
    products: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    missing: list[str] = []

    golden_dir = args.output_dir / "ttbar_puppi128"
    paths = save_figure(plot_golden_summary(records), golden_dir / "golden_summary")
    copied_manifest = golden_dir / "manifest.json"
    if args.golden_manifest.resolve() != copied_manifest.resolve():
        shutil.copy2(args.golden_manifest, copied_manifest)
    products.append({"label": "Golden MSE, utilization, and rate-distortion summary", "paths": paths + [golden_dir / "manifest.json"]})
    absent_golden_metrics = missing_golden_metric_records(records)
    if absent_golden_metrics:
        missing.append(
            "golden summary metrics for: " + ", ".join(absent_golden_metrics)
        )
    for row in records:
        provenance.append(_provenance_row("ttbar_puppi128/golden_summary", {"experiment": row.get("label"), "wandb_train_id": row.get("job"), "quantizer": row.get("plot_family"), "codebook_size": row.get("total_codebook_size"), "selection": "PUPPIW > 0.05, leading 128", "training_sample": "ttbar"}, test_sample="ttbar", events=10000))

    try:
        sweep_records = select_vq_ste_size_records(
            records, require_complete=not args.allow_incomplete
        )
        paths = save_figure(
            plot_vq_size_feature_sweep(
                records, require_complete=not args.allow_incomplete
            ),
            golden_dir / "vq_ste_feature_size_sweep",
        )
        sweep_values = {}
        for row in sweep_records:
            size = int(row["total_codebook_size"])
            for key, values in _load_npz(Path(row["histogram_path"])).items():
                if key.startswith("physical_"):
                    sweep_values[f"codes_{size}_{key}"] = values
        sweep_cache = golden_dir / "vq_ste_feature_size_sweep_histograms.npz"
        np.savez_compressed(sweep_cache, **sweep_values)
        products.append({"label": "VQ-STE feature and residual size sweep", "paths": paths + [sweep_cache]})
        for row in sweep_records:
            provenance.append(_provenance_row("ttbar_puppi128/vq_ste_feature_size_sweep", {"experiment": row["label"], "wandb_train_id": row.get("job"), "quantizer": "VQ-STE", "codebook_size": row["total_codebook_size"], "selection": "PUPPIW > 0.05, leading 128", "training_sample": "ttbar"}, test_sample="ttbar", events=10000))
        missing_sizes = sorted(
            set(VQ_STE_SIZES)
            - {int(row["total_codebook_size"]) for row in sweep_records}
        )
        if missing_sizes:
            missing.append(
                f"Missing VQ-STE histogram caches for codebook sizes: {missing_sizes}"
            )
    except (FileNotFoundError, ValueError) as exc:
        missing.append(str(exc))

    golden_4096_records = [
        row
        for row in records
        if row.get("plot_family") == "VQ STE" and row.get("total_codebook_size") == 4096
    ]
    golden_4096 = golden_4096_records[0] if len(golden_4096_records) == 1 else {}
    golden_4096_model = {
        "experiment": golden_4096.get("label", "ttbar VQ-STE 4096"),
        "wandb_train_id": golden_4096.get("job"),
        "checkpoint": str(Path(golden_4096["run_dir"]) / "checkpoints/best.ckpt")
        if golden_4096.get("run_dir")
        else None,
        "quantizer": "VQ-STE",
        "codebook_size": 4096,
        "selection": "PUPPIW > 0.05, leading 128",
        "training_sample": "ttbar",
    }
    if args.golden_pid_confusion and args.golden_pid_confusion.is_file():
        figure, counts, normalized = plot_pid_confusion(args.golden_pid_confusion)
        paths = save_figure(figure, golden_dir / "vq_ste_4096_pid_confusion")
        np.savez_compressed(golden_dir / "vq_ste_4096_pid_confusion.npz", counts=counts, row_normalized=normalized)
        products.append({"label": "VQ-STE 4096 PID confusion", "paths": paths + [golden_dir / "vq_ste_4096_pid_confusion.npz"]})
        provenance.append(_provenance_row("ttbar_puppi128/vq_ste_4096_pid_confusion", golden_4096_model, test_sample="ttbar", events=10000))
    else:
        missing.append("ttbar VQ-STE 4096 PID confusion JSON (--golden-pid-confusion)")
    if args.golden_pid_residuals and args.golden_pid_residuals.is_file():
        paths = save_figure(plot_pid_residuals(args.golden_pid_residuals), golden_dir / "vq_ste_4096_pid_residuals")
        copied = golden_dir / "vq_ste_4096_pid_residual_values.npz"
        shutil.copy2(args.golden_pid_residuals, copied)
        products.append({"label": "VQ-STE 4096 PID-conditioned residuals", "paths": paths + [copied]})
        provenance.append(_provenance_row("ttbar_puppi128/vq_ste_4096_pid_residuals", golden_4096_model, test_sample="ttbar", events=10000))
    else:
        missing.append("ttbar VQ-STE 4096 PID residual NPZ (--golden-pid-residuals)")

    models_dir = args.output_dir / "models_5m"
    metrics = discover_campaign_metrics(args.campaign_summary_root, _parse_explicit_metrics(args.campaign_metric))
    ablation_rows = write_ablation_table(metrics, models_dir / "ablation")
    table_paths = sorted((models_dir / "ablation").glob("particle_reconstruction_ablation.*"))
    products.append({"label": "Eight-run particle reconstruction ablation", "paths": table_paths})
    absent_metrics = [
        row["label"] for row in ablation_rows if row["reconstruction_mse"] is None
    ]
    if absent_metrics:
        missing.append("SM-test MSE for: " + ", ".join(absent_metrics))
    for row in CAMPAIGN_MODELS:
        provenance.append(_provenance_row("models_5m/ablation", row, test_sample="balanced SM mixture", events="complete suite"))

    downstream = dict(CAMPAIGN_MODELS[1])
    if args.downstream_run_dir:
        downstream.update(validate_downstream_run(args.downstream_run_dir.resolve()))
    downstream_dir = models_dir / "downstream" / f"vq_ste_{downstream['codebook_size']}_sm"
    downstream_specs = (
        ("higgs", args.higgs_cache, plot_higgs, "higgs_njets_and_mass", "ggHbb", "anti-kt R=0.4, pT >= 30 GeV", "two leading-pT jets"),
        ("tau32", args.tau32_cache, plot_tau32, "tau32_njets_and_shape", "all-hadronic ttbar", "anti-kt R=0.8, pT >= 450 GeV", "Hungarian delta-R < 0.4"),
    )
    for label, cache, renderer, filename, test_sample, jet_definition, matching in downstream_specs:
        if cache and cache.is_file():
            figure, values = renderer(cache)
            paths = save_figure(figure, downstream_dir / filename)
            copied = downstream_dir / f"{filename}_values.npz"
            np.savez_compressed(copied, **{key: value for key, value in values.items() if isinstance(value, np.ndarray)})
            products.append({"label": label, "paths": paths + [copied]})
            provenance.append(_provenance_row(f"models_5m/downstream/{filename}", downstream, test_sample=test_sample, events=10000, jet_definition=jet_definition, matching=matching))
        else:
            missing.append(
                f"{label} cache (--{label.lower() if label != 'tau32' else 'tau32'}-cache)"
            )
    met_model = dict(
        next(
            model
            for model in CAMPAIGN_MODELS
            if model["experiment"] == "orbit_campaign_full500_sm_vq8192_5m"
        )
    )
    if args.met_run_dir:
        met_model.update(validate_downstream_run(args.met_run_dir.resolve()))
    met_dir = models_dir / "downstream" / f"vq_ste_{met_model['codebook_size']}_sm"
    if args.met_cache and args.met_cache.is_file():
        figure, values = plot_met(args.met_cache)
        filename = "missing_transverse_energy"
        paths = save_figure(figure, met_dir / filename)
        copied = met_dir / f"{filename}_values.npz"
        np.savez_compressed(
            copied,
            **{key: value for key, value in values.items() if isinstance(value, np.ndarray)},
        )
        products.append({"label": "Missing transverse energy", "paths": paths + [copied]})
        provenance.append(
            _provenance_row(
                f"models_5m/downstream/{filename}",
                met_model,
                test_sample="balanced SM mixture",
                events=10000,
            )
        )
    else:
        missing.append("MET cache (--met-cache)")
    if args.jet_residual_cache and args.jet_residual_cache.is_file():
        copied = downstream_dir / "jet_residual_values.npz"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.jet_residual_cache, copied)
        for prefix, filename, label in (("all", "jet_residuals_all", "All-jet residuals"), ("leading5", "jet_residuals_leading5", "Leading-five jet residuals")):
            figure, _ = plot_jet_residuals(args.jet_residual_cache, prefix)
            paths = save_figure(figure, downstream_dir / filename)
            products.append({"label": label, "paths": paths + [copied]})
            provenance.append(_provenance_row(f"models_5m/downstream/{filename}", downstream, test_sample="balanced SM mixture", events=10000, jet_definition="anti-kt R=0.4, pT >= 30 GeV", matching="Hungarian delta-R < 0.2"))
    else:
        missing.append("SM-mixture jet-residual NPZ (--jet-residual-cache)")
    if args.sm_pid_residuals and args.sm_pid_residuals.is_file():
        paths = save_figure(plot_pid_residuals(args.sm_pid_residuals), downstream_dir / "pid_residuals")
        copied = downstream_dir / "pid_residual_values.npz"
        shutil.copy2(args.sm_pid_residuals, copied)
        products.append({"label": "SM PID-conditioned residuals", "paths": paths + [copied]})
        provenance.append(_provenance_row("models_5m/downstream/pid_residuals", downstream, test_sample="balanced SM mixture", events=10000))
    else:
        missing.append("SM-mixture PID residual NPZ (--sm-pid-residuals)")

    if args.comparison_dir:
        # Keep the spelling supplied by the caller: EOS home paths may resolve
        # through /eos/home-y while the collection is addressed via /eos/user/y.
        comparison_dir = args.comparison_dir
        comparison_specs = (
            ("5M-versus-200k ttbar particle residual comparison", "particle_residuals_vq_ste_4096"),
            ("5M-versus-200k ttbar MSE, utilization, and rate comparison", "vq_ste_training_scale_summary"),
        )
        for label, stem in comparison_specs:
            paths = [comparison_dir / f"{stem}.{suffix}" for suffix in ("pdf", "png")]
            if all(path.is_file() for path in paths):
                products.append({"label": label, "paths": paths})
            else:
                missing.append(f"comparison products for {stem} (--comparison-dir)")

    write_model_index(provenance, args.output_dir)
    write_readme(args.output_dir, products, missing)
    if missing and not args.allow_incomplete:
        raise RuntimeError("Writeup collection is incomplete:\n- " + "\n- ".join(missing))
    print(f"Wrote {len(products)} products to {args.output_dir}")
    if missing:
        print("Pending inputs:\n- " + "\n- ".join(missing))


if __name__ == "__main__":
    main()
