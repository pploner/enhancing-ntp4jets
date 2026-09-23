#!/usr/bin/env python
"""Build title-free canonical-tt presentation plots from cached evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from gabbro.plotting.orbit import (
    HISTOGRAM_FILL_ALPHA,
    HISTOGRAM_LINEWIDTH,
    MAX_CODEBOOK_MARKER_AREA,
    MIN_CODEBOOK_MARKER_AREA,
    ORIGINAL_COLOR,
    PRESENTATION_CODEBOOK_MARKER_AREA_SCALE,
    RECONSTRUCTED_COLOR,
    codebook_marker_areas,
    multirun_color,
    multirun_display_label,
    multirun_legend_sort_key,
    multirun_marker,
)


PRESENTATION_LEGEND_FONTSIZE = 12
PRESENTATION_LEGEND_MARKERSCALE = 1.35
HIGGS_MASS_PLOT_BINS = np.arange(40.0, 205.0, 5.0)
HIGGS_MASS_EXTENDED_PLOT_BINS = np.arange(40.0, 505.0, 5.0)
HIGGS_MASS_LOG_PLOT_BINS = np.arange(40.0, 1005.0, 5.0)
from gabbro.plotting.utils import set_mpl_style
from scripts.collect_orbit_multirun import _load_histograms, _save_figures


def double_crystal_ball(x, norm, mean, sigma, alpha_l, n_l, alpha_r, n_r):
    """Double-sided Crystal Ball model used by the Higgs evaluator."""
    t = (x - mean) / sigma
    result = np.exp(-0.5 * t**2)
    left = t < -alpha_l
    right = t > alpha_r
    a_l = (n_l / alpha_l) ** n_l * np.exp(-0.5 * alpha_l**2)
    b_l = n_l / alpha_l - alpha_l
    a_r = (n_r / alpha_r) ** n_r * np.exp(-0.5 * alpha_r**2)
    b_r = n_r / alpha_r - alpha_r
    result[left] = a_l * (b_l - t[left]) ** (-n_l)
    result[right] = a_r * (b_r + t[right]) ** (-n_r)
    return norm * result


def fit_mass(values: np.ndarray) -> dict:
    """Fit the same 40--200 GeV peak model as the Higgs evaluator."""
    values = np.asarray(values, dtype=np.float64)
    values = values[(values >= 40.0) & (values <= 200.0)]
    counts, edges = np.histogram(values, bins=np.arange(40.0, 202.0, 2.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(values) < 50:
        return {"success": False, "events": int(len(values)), "mean": None, "sigma": None}
    initial = [max(counts.max(), 1), np.median(values), max(np.std(values), 5.0), 1.5, 3, 1.5, 3]
    try:
        parameters, _ = curve_fit(
            double_crystal_ball,
            centers,
            counts,
            p0=initial,
            bounds=([0, 80, 1, 0.2, 1.01, 0.2, 1.01], [np.inf, 170, 80, 8, 30, 8, 30]),
            maxfev=50_000,
        )
    except (RuntimeError, ValueError):
        return {"success": False, "events": int(len(values)), "mean": None, "sigma": None}
    return {
        "success": True,
        "events": int(len(values)),
        "mean": float(parameters[1]),
        "sigma": float(parameters[2]),
        "parameters": list(map(float, parameters)),
    }


def bootstrap_fit(values: np.ndarray, replicas: int) -> dict:
    """Return deterministic bootstrap errors for fitted peak mean and width."""
    if len(values) < 50 or replicas < 1:
        return {"successful_replicas": 0, "mean_uncertainty": None, "sigma_uncertainty": None}
    rng = np.random.default_rng(12345)
    fits = [fit_mass(rng.choice(values, size=len(values), replace=True)) for _ in range(replicas)]
    means = [fit["mean"] for fit in fits if fit["success"]]
    sigmas = [fit["sigma"] for fit in fits if fit["success"]]
    return {
        "successful_replicas": len(means),
        "mean_uncertainty": float(np.std(means, ddof=1)) if len(means) > 1 else None,
        "sigma_uncertainty": float(np.std(sigmas, ddof=1)) if len(sigmas) > 1 else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-collection",
        nargs=2,
        action="append",
        metavar=("NAME", "MANIFEST"),
        required=True,
        help="Presentation subdirectory name and an existing multirun manifest.json.",
    )
    parser.add_argument("--higgs-full-manifest", type=Path, required=True)
    parser.add_argument("--higgs-near-manifest", type=Path, required=True)
    parser.add_argument("--higgs-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--higgs-observable-cache-dir",
        type=Path,
        help="Optional focused cache containing the near-4096 event observables.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicas", type=int, default=50)
    parser.add_argument("--original-bits-per-input-particle", type=float, default=43.0)
    parser.add_argument("--compression-reference-bits", type=float, default=40.0)
    parser.add_argument("--wandb-project", default="orbit-tokenizer")
    parser.add_argument("--wandb-name", default="canonical_tt_presentation_plots")
    parser.add_argument("--wandb-group", default="presentation_plots")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty record list in {path}")
    return records


def select_plot_families(records: list[dict], families: set[str]) -> list[dict]:
    """Select records belonging to the requested presentation families."""
    return [record for record in records if record.get("plot_family") in families]


def regenerate_golden_collection(
    name: str,
    manifest_path: Path,
    output_root: Path,
    original_bits_per_input_particle: float,
    compression_reference_bits: float,
) -> None:
    records = load_manifest(manifest_path)
    marker_areas = (
        PRESENTATION_CODEBOOK_MARKER_AREA_SCALE * codebook_marker_areas(records)
    )
    records = [
        {**record, "plot_marker_area": float(marker_area)}
        for record, marker_area in zip(records, marker_areas)
    ]
    histogram_runs = []
    for record in records:
        histogram_path = record.get("histogram_path")
        histograms = _load_histograms(Path(histogram_path)) if histogram_path else None
        if histograms is not None:
            family = record.get("plot_family") or record["label"]
            histogram_runs.append(
                {
                    "label": record["label"],
                    "plot_color": multirun_color(family),
                    "histograms": histograms,
                }
            )
    destination = output_root / "golden" / name
    destination.mkdir(parents=True, exist_ok=True)
    _save_figures(
        records,
        histogram_runs,
        destination,
        original_bits_per_input_particle=original_bits_per_input_particle,
        compression_reference_bits=compression_reference_bits,
        show_titles=False,
    )
    shutil.copy2(manifest_path, destination / "manifest.json")
    source_summary = manifest_path.with_name("summary.csv")
    if source_summary.is_file():
        shutil.copy2(source_summary, destination / "summary.csv")


def mass_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    original, decoded = [], []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            original.append(float(row["resolved_original"]))
            decoded.append(float(row["resolved_decoded"]))
    return np.asarray(original), np.asarray(decoded)


def observable_values(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an original/decoded event-level observable from a Higgs cache."""
    original, decoded = [], []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {f"{name}_original", f"{name}_decoded"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Higgs cache {path} predates {name}; missing columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            original.append(float(row[f"{name}_original"]))
            decoded.append(float(row[f"{name}_decoded"]))
    return np.asarray(original), np.asarray(decoded)


def fit_with_bootstrap(values: np.ndarray, replicas: int) -> dict:
    finite = values[np.isfinite(values)]
    return fit_mass(finite) | bootstrap_fit(finite, replicas)


def load_higgs_records(
    full_manifest: list[dict], cache_dir: Path, replicas: int
) -> tuple[list[dict], np.ndarray, dict]:
    results = []
    reference_values = None
    for index, metadata in enumerate(full_manifest):
        candidates = cache_dir / f"{index:03d}" / "higgs_candidates.csv"
        if not candidates.is_file():
            raise FileNotFoundError(f"Missing Higgs cache for manifest row {index}: {candidates}")
        original, decoded = mass_values(candidates)
        if reference_values is None:
            reference_values = original
        elif len(original) != len(reference_values) or not np.allclose(
            original, reference_values, equal_nan=True
        ):
            raise ValueError(
                f"Original Higgs candidates in cache {index:03d} do not match the shared reference"
            )
        fit = fit_with_bootstrap(decoded, replicas)
        results.append(
            {
                "index": index,
                "label": metadata["label"],
                "plot_family": metadata.get("plot_family") or metadata["label"],
                "run_dir": metadata["run_dir"],
                "total_codebook_size": metadata.get("total_codebook_size"),
                "marginal_bits_per_input_particle": metadata.get(
                    "metrics/rate/marginal_bits_per_input_particle"
                ),
                "decoded_efficiency": float(np.mean(np.isfinite(decoded))),
                **{f"fit_{key}": value for key, value in fit.items()},
            }
        )
    if reference_values is None:
        raise ValueError("No Higgs caches were loaded")
    return results, reference_values, fit_with_bootstrap(reference_values, replicas)


def plot_mu_sigma(records: list[dict], reference: dict, output_path: Path) -> None:
    set_mpl_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    grouped = defaultdict(list)
    for record in records:
        if record.get("fit_success"):
            grouped[record["plot_family"]].append(record)
    all_codebook_sizes = np.asarray(
        [record.get("total_codebook_size") or np.nan for record in records], dtype=float
    )
    finite_sizes = all_codebook_sizes[
        np.isfinite(all_codebook_sizes) & (all_codebook_sizes > 0)
    ]
    codebook_size_range = (float(finite_sizes.min()), float(finite_sizes.max()))
    for family, family_records in sorted(
        grouped.items(), key=lambda item: multirun_legend_sort_key(item[0])
    ):
        ordered = sorted(
            family_records,
            key=lambda record: (record.get("total_codebook_size") or 0, record["label"]),
        )
        means = np.asarray([record["fit_mean"] for record in ordered])
        sigmas = np.asarray([record["fit_sigma"] for record in ordered])
        color = multirun_color(family)
        if len(ordered) > 1:
            axis.plot(means, sigmas, color=color, linewidth=1.5, alpha=0.75)
        axis.scatter(
            means,
            sigmas,
            s=(
                PRESENTATION_CODEBOOK_MARKER_AREA_SCALE
                * codebook_marker_areas(ordered, codebook_size_range)
            ),
            color=color,
            marker=multirun_marker(family),
            label=multirun_display_label(family),
            zorder=3,
        )
    if reference.get("success"):
        mean, sigma = reference["mean"], reference["sigma"]
        axis.axvline(mean, color="black", linestyle="--", alpha=0.35)
        axis.axhline(sigma, color="black", linestyle="--", alpha=0.35)
        axis.scatter(
            [mean],
            [sigma],
            s=240,
            color="black",
            marker="*",
            label="Original",
            zorder=5,
        )
    axis.set_xlabel(r"Fitted Higgs mass peak $\mu$ [GeV]")
    axis.set_ylabel(r"Fitted Higgs mass width $\sigma$ [GeV]", fontsize=20)
    axis.set_ylim(top=max(axis.get_ylim()[1], 42.0))
    axis.legend(
        fontsize=PRESENTATION_LEGEND_FONTSIZE,
        markerscale=PRESENTATION_LEGEND_MARKERSCALE,
        loc="upper right",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_mass_response(
    records: list[dict],
    reference: dict,
    output_path: Path,
    compression_reference_bits: float = 40.0,
) -> None:
    """Plot fitted Higgs peak response against marginal compression rate."""
    if not reference.get("success") or not reference.get("mean"):
        raise ValueError("A successful non-zero original Higgs fit is required")
    usable = [
        record
        for record in records
        if record.get("fit_success")
        and record.get("marginal_bits_per_input_particle") is not None
    ]
    if not usable:
        raise ValueError("No fitted Higgs records with marginal-rate information")

    set_mpl_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    grouped = defaultdict(list)
    for record in usable:
        grouped[record["plot_family"]].append(record)
    finite_sizes = np.asarray(
        [record.get("total_codebook_size") or np.nan for record in usable], dtype=float
    )
    finite_sizes = finite_sizes[np.isfinite(finite_sizes) & (finite_sizes > 0)]
    codebook_size_range = (float(finite_sizes.min()), float(finite_sizes.max()))
    original_peak = float(reference["mean"])

    for family, family_records in sorted(
        grouped.items(), key=lambda item: multirun_legend_sort_key(item[0])
    ):
        ordered = sorted(
            family_records,
            key=lambda record: (record.get("total_codebook_size") or 0, record["label"]),
        )
        compression_ratios = np.asarray(
            [
                float(record["marginal_bits_per_input_particle"])
                / compression_reference_bits
                for record in ordered
            ]
        )
        mass_responses = np.asarray(
            [float(record["fit_mean"]) / original_peak for record in ordered]
        )
        color = multirun_color(family)
        if len(ordered) > 1:
            axis.plot(
                compression_ratios,
                mass_responses,
                color=color,
                linewidth=1.5,
                alpha=0.75,
            )
        axis.scatter(
            compression_ratios,
            mass_responses,
            s=(
                PRESENTATION_CODEBOOK_MARKER_AREA_SCALE
                * codebook_marker_areas(ordered, codebook_size_range)
            ),
            color=color,
            marker=multirun_marker(family),
            label=multirun_display_label(family),
            zorder=3,
        )

    axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Original",
    )
    axis.set_xlabel(f"Compressed rate / {compression_reference_bits:g}-bit input payload")
    axis.set_ylabel(r"$m_{\mathrm{H}}^{\mathrm{reco}}/m_{\mathrm{H}}^{\mathrm{original}}$")
    axis.legend(
        fontsize=PRESENTATION_LEGEND_FONTSIZE,
        markerscale=PRESENTATION_LEGEND_MARKERSCALE,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_near_histogram(
    near_manifest: list[dict],
    full_manifest: list[dict],
    cache_dir: Path,
    reference_values: np.ndarray,
    replicas: int,
    output_path: Path,
    bins: np.ndarray = HIGGS_MASS_PLOT_BINS,
    log_y: bool = False,
) -> None:
    set_mpl_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    original = reference_values[np.isfinite(reference_values)]
    original_fit = fit_with_bootstrap(original, replicas)
    axis.hist(
        original,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color="black",
        alpha=0.25,
        linewidth=1.8,
        label=(
            f"Original ($\\mu$={original_fit['mean']:.1f}, "
            f"$\\sigma$={original_fit['sigma']:.1f} GeV)"
        ),
    )
    full_by_run = {
        str(record["run_dir"]): (index, record)
        for index, record in enumerate(full_manifest)
    }
    for near in sorted(
        near_manifest,
        key=lambda record: multirun_legend_sort_key(
            record.get("plot_family") or record["label"]
        ),
    ):
        key = str(near["run_dir"])
        if key not in full_by_run:
            raise ValueError(f"Near-4096 run is absent from the full manifest: {key}")
        index, full = full_by_run[key]
        _, decoded = mass_values(cache_dir / f"{index:03d}" / "higgs_candidates.csv")
        decoded = decoded[np.isfinite(decoded)]
        fit = fit_with_bootstrap(decoded, replicas)
        label = multirun_display_label(near.get("plot_family") or near["label"])
        if fit.get("success"):
            label += f" ($\\mu$={fit['mean']:.1f}, $\\sigma$={fit['sigma']:.1f} GeV)"
        axis.hist(
            decoded,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=multirun_color(full.get("plot_family") or full["label"]),
            label=label,
        )
    axis.set_xlabel("Higgs candidate mass [GeV]")
    axis.set_ylabel("Normalized events")
    if log_y:
        axis.set_yscale("log")
    axis.legend(
        fontsize=PRESENTATION_LEGEND_FONTSIZE,
        markerscale=PRESENTATION_LEGEND_MARKERSCALE,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_single_mass_histogram(
    cache_path: Path,
    replicas: int,
    output_path: Path,
    bins: np.ndarray = HIGGS_MASS_PLOT_BINS,
    log_y: bool = False,
) -> None:
    """Plot one cached Higgs candidate distribution in presentation style."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original, decoded = mass_values(cache_path)
    common = np.isfinite(original) & np.isfinite(decoded)
    original = original[common]
    decoded = decoded[common]
    original_fit = fit_with_bootstrap(original, replicas)
    decoded_fit = fit_with_bootstrap(decoded, replicas)

    set_mpl_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.hist(
        original,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=ORIGINAL_COLOR,
        alpha=HISTOGRAM_FILL_ALPHA,
        linewidth=HISTOGRAM_LINEWIDTH,
        label=(
            f"Original ($\\mu$={original_fit['mean']:.1f}, "
            f"$\\sigma$={original_fit['sigma']:.1f} GeV)"
        ),
    )
    axis.hist(
        decoded,
        bins=bins,
        density=True,
        histtype="step",
        color=RECONSTRUCTED_COLOR,
        linewidth=HISTOGRAM_LINEWIDTH,
        label=(
            f"Decoded ($\\mu$={decoded_fit['mean']:.1f}, "
            f"$\\sigma$={decoded_fit['sigma']:.1f} GeV)"
        ),
    )
    axis.set_xlabel("Higgs candidate mass [GeV]")
    axis.set_ylabel("Normalized events")
    if log_y:
        axis.set_yscale("log")
    axis.legend(
        fontsize=PRESENTATION_LEGEND_FONTSIZE,
        markerscale=PRESENTATION_LEGEND_MARKERSCALE,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_near_observable_histogram(
    near_manifest: list[dict],
    full_manifest: list[dict],
    cache_dir: Path,
    name: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """Overlay a cached event observable for the near-4096 model collection."""
    full_by_run = {
        str(record["run_dir"]): (index, record)
        for index, record in enumerate(full_manifest)
    }
    series = []
    reference = None
    for near in near_manifest:
        key = str(near["run_dir"])
        if key not in full_by_run:
            raise ValueError(f"Near-4096 run is absent from the full manifest: {key}")
        index, full = full_by_run[key]
        path = cache_dir / f"{index:03d}" / "higgs_candidates.csv"
        original, decoded = observable_values(path, name)
        if reference is None:
            reference = original
        elif len(reference) != len(original) or not np.allclose(
            reference, original, equal_nan=True
        ):
            raise ValueError(f"Original {name} values in cache {index:03d} differ")
        series.append((near, full, decoded))
    finite = [reference[np.isfinite(reference)]]
    finite.extend(values[np.isfinite(values)] for _, _, values in series)
    combined = np.concatenate([values for values in finite if len(values)])
    if name == "n_jets":
        upper = int(np.max(combined)) if len(combined) else 0
        bins = np.arange(-0.5, upper + 1.5, 1.0)
    else:
        upper = max(float(np.max(combined)) if len(combined) else 1.0, 1.0)
        bins = np.linspace(0.0, upper, 61)

    set_mpl_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.hist(
        reference,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color="black",
        alpha=0.25,
        linewidth=1.8,
        label="Original",
    )
    for near, full, decoded in sorted(
        series,
        key=lambda item: multirun_legend_sort_key(
            item[0].get("plot_family") or item[0]["label"]
        ),
    ):
        label = multirun_display_label(near.get("plot_family") or near["label"])
        axis.hist(
            decoded,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=multirun_color(full.get("plot_family") or full["label"]),
            label=label,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Normalized events")
    axis.set_yscale("log")
    axis.legend(
        fontsize=PRESENTATION_LEGEND_FONTSIZE,
        markerscale=PRESENTATION_LEGEND_MARKERSCALE,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_single_observable_histogram(
    cache_path: Path,
    name: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """Plot original and decoded observables for one model in presentation style."""
    original, decoded = observable_values(cache_path, name)
    finite = np.concatenate(
        (original[np.isfinite(original)], decoded[np.isfinite(decoded)])
    )
    if name == "n_jets":
        upper = int(np.max(finite)) if len(finite) else 0
        bins = np.arange(-0.5, upper + 1.5, 1.0)
    else:
        upper = max(float(np.max(finite)) if len(finite) else 1.0, 1.0)
        bins = np.linspace(0.0, upper, 61)

    set_mpl_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.hist(
        original,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=ORIGINAL_COLOR,
        alpha=HISTOGRAM_FILL_ALPHA,
        linewidth=HISTOGRAM_LINEWIDTH,
        label="Original",
    )
    axis.hist(
        decoded,
        bins=bins,
        density=True,
        histtype="step",
        color=RECONSTRUCTED_COLOR,
        linewidth=HISTOGRAM_LINEWIDTH,
        label="Decoded",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Normalized events")
    axis.set_yscale("log")
    axis.legend(
        fontsize=PRESENTATION_LEGEND_FONTSIZE,
        markerscale=PRESENTATION_LEGEND_MARKERSCALE,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_higgs_summary(records: list[dict], reference: dict, output_dir: Path) -> None:
    payload = {"reference": reference, "runs": records}
    (output_dir / "higgs_mass_fit_summary.json").write_text(json.dumps(payload, indent=2))
    fields = sorted({key for record in records for key in record})
    with (output_dir / "higgs_mass_fit_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def upload_to_wandb(args: argparse.Namespace) -> None:
    if args.no_wandb:
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B synchronization requested, but wandb is not installed") from exc
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        entity=args.wandb_entity,
        job_type="presentation-plots",
        config={"bootstrap_replicas": args.bootstrap_replicas, "titles": False},
    )
    try:
        images = {
            f"presentation_plots/{path.relative_to(args.output_dir).with_suffix('')}": wandb.Image(
                str(path)
            )
            for path in sorted(args.output_dir.rglob("*.png"))
        }
        if images:
            run.log(images)
        artifact = wandb.Artifact(
            f"canonical-tt-presentation-plots-{run.id}",
            type="presentation-plots",
        )
        for suffix in ("*.png", "*.json", "*.csv"):
            for path in sorted(args.output_dir.rglob(suffix)):
                if "higgs_cache" in path.relative_to(args.output_dir).parts:
                    continue
                artifact.add_file(str(path), name=str(path.relative_to(args.output_dir)))
        run.log_artifact(artifact)
    finally:
        wandb.finish()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicas < 1:
        raise ValueError("--bootstrap-replicas must be positive")
    if args.compression_reference_bits <= 0:
        raise ValueError("--compression-reference-bits must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in args.golden_collection:
        regenerate_golden_collection(
            name,
            Path(manifest),
            args.output_dir,
            args.original_bits_per_input_particle,
            args.compression_reference_bits,
        )
    full_manifest = load_manifest(args.higgs_full_manifest)
    near_manifest = load_manifest(args.higgs_near_manifest)
    if len(full_manifest) != 46:
        raise ValueError(f"Expected 46 full-sweep runs, found {len(full_manifest)}")
    if len(near_manifest) != 6:
        raise ValueError(f"Expected 6 near-4096 runs, found {len(near_manifest)}")
    higgs_dir = args.output_dir / "higgs"
    higgs_dir.mkdir(parents=True, exist_ok=True)
    records, reference_values, reference_fit = load_higgs_records(
        full_manifest, args.higgs_cache_dir, args.bootstrap_replicas
    )
    plot_mu_sigma(records, reference_fit, higgs_dir / "higgs_mass_mu_vs_sigma.png")
    plot_mass_response(
        records,
        reference_fit,
        higgs_dir / "higgs_mass_response_vs_compression_ratio.png",
        args.compression_reference_bits,
    )
    faiss_vq_ste_records = select_plot_families(
        records, {"FAISS k-means", "VQ STE"}
    )
    plot_mu_sigma(
        faiss_vq_ste_records,
        reference_fit,
        higgs_dir / "higgs_mass_mu_vs_sigma_faiss_vq_ste.png",
    )
    plot_mass_response(
        faiss_vq_ste_records,
        reference_fit,
        higgs_dir / "higgs_mass_response_vs_compression_ratio_faiss_vq_ste.png",
        args.compression_reference_bits,
    )
    plot_near_histogram(
        near_manifest,
        full_manifest,
        args.higgs_cache_dir,
        reference_values,
        args.bootstrap_replicas,
        higgs_dir / "higgs_mass_near_4096.png",
    )
    plot_near_histogram(
        near_manifest,
        full_manifest,
        args.higgs_cache_dir,
        reference_values,
        args.bootstrap_replicas,
        higgs_dir / "higgs_mass_near_4096_extended_500.png",
        bins=HIGGS_MASS_EXTENDED_PLOT_BINS,
    )
    plot_near_histogram(
        near_manifest,
        full_manifest,
        args.higgs_cache_dir,
        reference_values,
        args.bootstrap_replicas,
        higgs_dir / "higgs_mass_near_4096_log_1000.png",
        bins=HIGGS_MASS_LOG_PLOT_BINS,
        log_y=True,
    )
    plot_near_observable_histogram(
        near_manifest,
        full_manifest,
        args.higgs_observable_cache_dir or args.higgs_cache_dir,
        "leading_particle_pt",
        r"Leading particle $p_{\mathrm{T}}$ [GeV]",
        higgs_dir / "leading_particle_pt_near_4096.png",
    )
    plot_near_observable_histogram(
        near_manifest,
        full_manifest,
        args.higgs_observable_cache_dir or args.higgs_cache_dir,
        "n_jets",
        "Number of AK4 jets",
        higgs_dir / "n_jets_near_4096.png",
    )
    vq_ste = [
        record
        for record in near_manifest
        if "vq_ste_codes_4096" in str(record.get("label", ""))
    ]
    if len(vq_ste) != 1:
        raise ValueError(f"Expected one VQ-STE 4096 record, found {len(vq_ste)}")
    full_by_run = {
        str(record["run_dir"]): index for index, record in enumerate(full_manifest)
    }
    vq_ste_index = full_by_run[str(vq_ste[0]["run_dir"])]
    vq_ste_cache = (
        (args.higgs_observable_cache_dir or args.higgs_cache_dir)
        / f"{vq_ste_index:03d}"
        / "higgs_candidates.csv"
    )
    vq_ste_mass_cache = (
        args.higgs_cache_dir / f"{vq_ste_index:03d}" / "higgs_candidates.csv"
    )
    single_model_dir = args.output_dir / "single_models" / "vq_ste_4096_tt"
    plot_single_mass_histogram(
        vq_ste_mass_cache,
        args.bootstrap_replicas,
        single_model_dir / "higgs_mass_resolved.png",
    )
    plot_single_mass_histogram(
        vq_ste_mass_cache,
        args.bootstrap_replicas,
        single_model_dir / "higgs_mass_resolved_extended_500.png",
        bins=HIGGS_MASS_EXTENDED_PLOT_BINS,
    )
    plot_single_mass_histogram(
        vq_ste_mass_cache,
        args.bootstrap_replicas,
        single_model_dir / "higgs_mass_resolved_log_1000.png",
        bins=HIGGS_MASS_LOG_PLOT_BINS,
        log_y=True,
    )
    plot_single_observable_histogram(
        vq_ste_cache,
        "leading_particle_pt",
        r"Leading particle $p_{\mathrm{T}}$ [GeV]",
        higgs_dir / "leading_particle_pt_vq_ste_4096.png",
    )
    plot_single_observable_histogram(
        vq_ste_cache,
        "n_jets",
        "Number of AK4 jets",
        higgs_dir / "n_jets_vq_ste_4096.png",
    )
    write_higgs_summary(records, reference_fit, higgs_dir)
    upload_to_wandb(args)
    print(f"Wrote canonical-tt presentation plots to {args.output_dir}")


if __name__ == "__main__":
    main()
