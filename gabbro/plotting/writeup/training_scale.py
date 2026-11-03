#!/usr/bin/env python
"""Compare matched 5M and 200k-event VQ-STE models."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

from gabbro.plotting.orbit import CODEBOOK_FAMILY_COLORS
from gabbro.plotting.writeup.summary import add_writeup_header, save_figure

COLOR = CODEBOOK_FAMILY_COLORS["vq_ste"]
VQ_STE_SIZES = {128, 256, 512, 1024, 2048, 4096, 8192, 16384}
SELECTIONS = {
    "full500": {
        "sequence_type": "particle_full",
        "max_sequence_length": 500,
        "label": r"full-$500$",
        "header": r"Unfiltered leading-$p_T$ particles  |  $N_{\max}=500$",
    },
    "puppi128": {
        "sequence_type": "particle",
        "max_sequence_length": 128,
        "label": r"PUPPI-$128$",
        "header": (
            r"Pileup-subtracted  |  $w_{\mathrm{PUPPI}}>0.05$  |  "
            r"$N_{\max}=128$  |  batch 256, no accumulation"
        ),
    },
}


def _nested(payload: dict, *keys: str):
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _find_run_config(path: Path) -> tuple[Path, dict]:
    """Locate a run config from either a run directory or one of its artifacts."""
    start = path if path.is_dir() else path.parent
    candidates = ("config_resolved.yaml", "config.yaml", ".hydra/config.yaml")
    for directory in (start, *list(start.parents)[:6]):
        for relative in candidates:
            candidate = directory / relative
            if candidate.is_file():
                cfg = yaml.safe_load(candidate.read_text())
                # W&B exports wrap each top-level setting in ``value`` and
                # serialize nested Hydra dictionaries as Python literals.
                if isinstance(cfg, dict) and isinstance(cfg.get("data"), dict) and "value" in cfg["data"]:
                    unwrapped = {}
                    for key, entry in cfg.items():
                        value = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
                        if isinstance(value, str) and value.lstrip().startswith("{"):
                            try:
                                value = ast.literal_eval(value)
                            except (SyntaxError, ValueError):
                                pass
                        unwrapped[key] = value
                    cfg = unwrapped
                return candidate, cfg
    raise FileNotFoundError(f"No run config found above {path}")


def validate_run(
    path: Path,
    *,
    selection: str,
    expected_train_events: int,
    expected_codes: int | None = None,
    expected_train_groups: set[str] | None = None,
) -> tuple[Path, dict]:
    """Validate the input selection and return normalized run metadata."""
    selection_spec = SELECTIONS[selection]
    config_path, cfg = _find_run_config(path)
    configured_length = _nested(cfg, "data", "max_sequence_length")
    # Legacy canonical particle configs left this null and used the schema
    # default of 128.  Normalize that implicit value for comparison.
    resolved_length = (
        128
        if configured_length is None
        and _nested(cfg, "data", "sequence_type") == "particle"
        else configured_length
    )
    observed = {
        "sequence_type": _nested(cfg, "data", "sequence_type"),
        "max_sequence_length": resolved_length,
        "train_event_budget": _nested(cfg, "data", "train_event_budget"),
        "num_codes": _nested(cfg, "model", "model_kwargs", "vq_kwargs", "num_codes"),
        "train_groups": sorted(
            {
                process.get("group")
                for process in (_nested(cfg, "data", "train_val_processes") or {}).values()
            }
        ),
    }
    expected = {
        "sequence_type": selection_spec["sequence_type"],
        "max_sequence_length": selection_spec["max_sequence_length"],
        "train_event_budget": expected_train_events,
    }
    if expected_codes is not None:
        expected["num_codes"] = expected_codes
    if expected_train_groups is not None:
        expected["train_groups"] = sorted(expected_train_groups)
    mismatches = {
        key: (observed[key], value)
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(
            f"Comparison input is not the requested {selection} run: "
            f"{details}; {config_path}"
        )
    metadata = {
        **observed,
        "batch_size": _nested(cfg, "data", "batch_size"),
        "accumulate_grad_batches": (
            _nested(cfg, "trainer", "accumulate_grad_batches") or 1
        ),
        "learning_rate": _nested(cfg, "model", "optimizer", "lr"),
        "scheduler": _nested(cfg, "model", "scheduler"),
    }
    return config_path, metadata


def validate_full500_run(
    path: Path,
    *,
    expected_train_events: int,
    expected_codes: int | None = None,
    expected_train_groups: set[str] | None = None,
) -> Path:
    """Backward-compatible validator retained for existing callers/tests."""
    config_path, _ = validate_run(
        path,
        selection="full500",
        expected_train_events=expected_train_events,
        expected_codes=expected_codes,
        expected_train_groups=expected_train_groups,
    )
    return config_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-manifest",
        "--golden-manifest",
        dest="canonical_manifest",
        type=Path,
        required=True,
        help="200k-event full-event ttbar VQ-STE manifest (the old option name is retained).",
    )
    parser.add_argument("--campaign-residual-cache", type=Path, required=True)
    parser.add_argument("--campaign-4096-metrics", type=Path, required=True)
    parser.add_argument("--canonical-4096-residual-cache", type=Path)
    parser.add_argument("--canonical-4096-metrics", type=Path)
    parser.add_argument(
        "--campaign-4096-config",
        type=Path,
        required=True,
        help="Resolved Hydra config or exported W&B config for the 5M ttbar model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=tuple(SELECTIONS),
        default="full500",
        help="Particle selection shared by the 200k- and 5M-event runs.",
    )
    parser.add_argument(
        "--training-domain",
        choices=("tt", "sm"),
        default="tt",
        help="Training domain shared by the 200k and 5M models.",
    )
    parser.add_argument(
        "--canonical-plot-family",
        default="VQ STE",
        help="Manifest family containing the 200k-reference comparison points.",
    )
    return parser.parse_args(argv)


def finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def metric(payload: dict, *names: str) -> float:
    for name in names:
        value = finite(payload.get(name))
        if value is not None:
            return value
    raise KeyError(f"None of {names!r} found in metrics payload")


def _residual_curve(histogram, feature: str) -> tuple[np.ndarray, np.ndarray]:
    canonical_prefix = f"physical_{feature}_diff"
    campaign_prefix = f"particle_{feature}_diff"
    for prefix in (canonical_prefix, campaign_prefix):
        counts_key = f"{prefix}_counts"
        bins_key = f"{prefix}_bins"
        if counts_key in histogram and bins_key in histogram:
            return histogram[counts_key], histogram[bins_key]
    raise KeyError(f"No residual histogram for {feature!r}")


def plot_residuals(
    canonical_hist: Path,
    campaign_hist: Path,
    selection: str,
    training_domain: str,
):
    hep.style.use("CMS")
    domain_label = r"$t\bar{t}$" if training_domain == "tt" else "SM mixture"
    selection_label = SELECTIONS[selection]["label"]
    canonical_label = rf"{domain_label} training, 200k events, {selection_label}"
    campaign_label = rf"{domain_label} training, 5M events, {selection_label}"
    canonical = np.load(canonical_hist)
    campaign = np.load(campaign_hist)
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), constrained_layout=True)
    specs = (
        ("pT", r"$p_\mathrm{T}^{\mathrm{dec}}-p_\mathrm{T}^{\mathrm{orig}}$ [GeV]"),
        ("Eta", r"$\eta^{\mathrm{dec}}-\eta^{\mathrm{orig}}$"),
        ("Phi", r"wrapped $\phi^{\mathrm{dec}}-\phi^{\mathrm{orig}}$"),
    )
    for ax, (feature, xlabel) in zip(axes, specs, strict=True):
        canonical_counts, canonical_bins = _residual_curve(canonical, feature)
        campaign_counts, campaign_bins = _residual_curve(campaign, feature)
        ax.stairs(
            canonical_counts, canonical_bins, color=COLOR, linestyle="-",
            linewidth=2.1, label=canonical_label,
        )
        ax.stairs(
            campaign_counts, campaign_bins, color=COLOR, linestyle="--",
            linewidth=2.4, label=campaign_label,
        )
        ax.axvline(0.0, color="0.35", linewidth=1.1, alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
    axes[0].legend(fontsize=13, frameon=False)
    add_writeup_header(
        figure,
        axes,
        training_text=rf"{domain_label} (200k and 5M), VQ-STE, 4096 codes",
        test_text=(
            r"$t\bar{t}$ (10k)" if training_domain == "tt" else "balanced SM mixture (10k)"
        ),
        selection_text=SELECTIONS[selection]["header"],
    )
    return figure


def plot_scale_summary(
    canonical: list[dict],
    campaign: list[dict],
    selection: str,
    training_domain: str,
):
    hep.style.use("CMS")
    domain_label = r"$t\bar{t}$" if training_domain == "tt" else "SM mixture"
    selection_label = SELECTIONS[selection]["label"]
    canonical_label = rf"{domain_label} training, 200k events, {selection_label}"
    campaign_label = rf"{domain_label} training, 5M events, {selection_label}"
    figure, axes = plt.subplots(1, 3, figsize=(20.5, 5.4), constrained_layout=True)
    panels = (
        ("codebook", "mse", "Nominal codebook size", "Reconstruction MSE"),
        ("codebook", "utilization", "Nominal codebook size", "Codebook utilization"),
        ("rate", "mse", "Compressed rate / 40-bit input", "Reconstruction MSE"),
    )
    for ax, (x_key, y_key, xlabel, ylabel) in zip(axes, panels, strict=True):
        for values, label, marker, linestyle, filled in (
            (canonical, canonical_label, "s", "-", False),
            (campaign, campaign_label, "o", "--", True),
        ):
            rows = sorted(values, key=lambda row: row[x_key])
            ax.plot(
                [row[x_key] for row in rows],
                [row[y_key] for row in rows],
                color=COLOR,
                marker=marker,
                markerfacecolor=COLOR if filled else "white",
                markeredgecolor=COLOR,
                markeredgewidth=1.8,
                markersize=8.5,
                linewidth=2.0,
                linestyle=linestyle,
                label=label,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        if x_key == "codebook":
            ax.set_xscale("log", base=2)
        if y_key == "mse":
            ax.set_yscale("log")
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=12.5, frameon=False)
    add_writeup_header(
        figure,
        axes,
        training_text=rf"{domain_label} (200k and 5M), VQ-STE",
        test_text=(
            r"$t\bar{t}$ (10k)" if training_domain == "tt" else "balanced SM mixture (10k)"
        ),
        selection_text=SELECTIONS[selection]["header"],
    )
    return figure


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    canonical_groups = (
        {"tt"} if args.training_domain == "tt" else {"QCD", "tt", "VJets", "VV"}
    )
    campaign_groups = canonical_groups if args.training_domain == "tt" else {
        "QCD",
        "tt",
        "VJets",
        "VV",
        "DY",
    }
    domain_plain = "tt" if args.training_domain == "tt" else "SM mixture"
    domain_label = r"$t\bar{t}$" if args.training_domain == "tt" else "SM mixture"
    records = json.loads(args.canonical_manifest.read_text())
    canonical = []
    canonical_4096 = None
    canonical_4096_metadata = None
    for record in records:
        if record.get("plot_family") != args.canonical_plot_family:
            continue
        codebook = finite(record.get("total_codebook_size"))
        if codebook is None:
            continue
        _, run_metadata = validate_run(
            Path(record["run_dir"]),
            selection=args.selection,
            expected_train_events=200_000,
            expected_codes=int(codebook),
            expected_train_groups=canonical_groups,
        )
        mse = finite(record.get("plot_metrics/reco_mse_total"))
        utilization = finite(record.get("metrics/utilization_total"))
        rate = finite(record.get("metrics/rate/marginal_bits_per_input_particle"))
        if None not in (codebook, mse, utilization, rate):
            canonical.append(
                {
                    "codebook": codebook,
                    "mse": mse,
                    "utilization": utilization,
                    "rate": rate / 40.0,
                }
            )
        if codebook == 4096:
            canonical_4096 = Path(record["histogram_path"])
            canonical_4096_metadata = run_metadata
    if not canonical or canonical_4096 is None:
        raise ValueError(
            "The 200k full-event manifest has no usable VQ-STE series or 4096 histogram"
        )

    if args.canonical_4096_residual_cache is not None:
        canonical_4096 = args.canonical_4096_residual_cache
    if args.canonical_4096_metrics is not None:
        canonical_metrics = json.loads(args.canonical_4096_metrics.read_text())
        canonical_active = finite(
            canonical_metrics.get("metrics/active_codes_total")
        )
        canonical = [
            {
                "codebook": 4096.0,
                "mse": metric(canonical_metrics, "metrics/mse_total"),
                "utilization": (
                    metric(canonical_metrics, "metrics/utilization_total")
                    if "metrics/utilization_total" in canonical_metrics
                    else canonical_active / 4096.0
                ),
                "rate": metric(
                    canonical_metrics,
                    "metrics/rate/marginal_bits_per_input_particle",
                )
                / 40.0,
            }
        ]

    campaign_4096_config, campaign_metadata = validate_run(
        args.campaign_4096_config,
        selection=args.selection,
        expected_train_events=5_000_000,
        expected_codes=4096,
        expected_train_groups=campaign_groups,
    )
    if args.selection == "puppi128":
        for label, metadata in (
            ("200k-event reference", canonical_4096_metadata),
            ("5M-event model", campaign_metadata),
        ):
            observed_batching = (
                metadata["batch_size"],
                metadata["accumulate_grad_batches"],
            )
            if observed_batching != (256, 1):
                raise ValueError(
                    f"{label} is not a batch-matched PUPPI-128 run: "
                    f"batch_size={observed_batching[0]!r}, "
                    f"accumulate_grad_batches={observed_batching[1]!r}"
                )

    metrics_4096 = json.loads(args.campaign_4096_metrics.read_text())
    active_codes = finite(metrics_4096.get("metrics/active_codes_total"))
    campaign = [
        {
            "codebook": 4096.0,
            "mse": metric(metrics_4096, "metrics/mse_total"),
            "utilization": (
                metric(metrics_4096, "metrics/utilization_total")
                if "metrics/utilization_total" in metrics_4096
                else active_codes / 4096.0
            ),
            "rate": metric(
                metrics_4096, "metrics/rate/marginal_bits_per_input_particle"
            ) / 40.0,
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = []
    products.extend(
        save_figure(
            plot_residuals(
                canonical_4096,
                args.campaign_residual_cache,
                args.selection,
                args.training_domain,
            ),
            args.output_dir / "particle_residuals_vq_ste_4096",
        )
    )
    products.extend(
        save_figure(
            plot_scale_summary(
                canonical,
                campaign,
                args.selection,
                args.training_domain,
            ),
            args.output_dir / "vq_ste_training_scale_summary",
        )
    )
    values = {
        "reference_200k": canonical,
        "models_5m": campaign,
        "selection": args.selection,
        "training_domain": args.training_domain,
        "labels": {
            "reference_200k": rf"{domain_label} training, 200k events, {SELECTIONS[args.selection]['label']}",
            "models_5m": rf"{domain_label} training, 5M events, {SELECTIONS[args.selection]['label']}",
        },
        "provenance": {
            "reference_manifest": str(args.canonical_manifest.resolve()),
            "reference_4096_histogram": str(canonical_4096.resolve()),
            "model_5m_residual_cache": str(args.campaign_residual_cache.resolve()),
            "model_5m_4096_metrics": str(args.campaign_4096_metrics.resolve()),
            "model_5m_4096_config": str(campaign_4096_config.resolve()),
            "reference_4096_run_metadata": canonical_4096_metadata,
            "model_5m_run_metadata": campaign_metadata,
        },
        "comparison_note": (
            f"Both highlighted 4096-code models are {domain_plain}-trained VQ-STE "
            f"models using the {args.selection} selection and are evaluated on the "
            f"same {domain_plain} test sample. For the "
            "PUPPI-128 comparison both use physical batches of 256 without gradient "
            "accumulation. "
            + (
                "The SM comparison is the closest available match: the 200k training "
                "sample groups DY within VJets, whereas the 5M training sample weights "
                "DY as a standalone fifth class."
                if args.training_domain == "sm"
                else "The training recipes still differ in learning rate and schedule."
            )
        ),
        "missing_reference_points": sorted(
            VQ_STE_SIZES - {int(row["codebook"]) for row in canonical}
        ),
    }
    values_path = args.output_dir / "comparison_values.json"
    values_path.write_text(json.dumps(values, indent=2) + "\n")
    products.append(values_path)
    readme = [
        "# 5M-versus-200k VQ-STE comparisons",
        "",
        values["comparison_note"],
        "Missing 200k-reference codebook sizes: "
        + (
            ", ".join(str(size) for size in values["missing_reference_points"])
            if values["missing_reference_points"]
            else "none"
        )
        + ".",
        "",
        *[f"- [{path.name}]({path.name})" for path in products],
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(f"Wrote {len(products)} products to {args.output_dir}")


if __name__ == "__main__":
    main()
