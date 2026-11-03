#!/usr/bin/env python
"""Render the joint eight-model 5M comparison from durable caches."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

from gabbro.plotting.orbit import ORIGINAL_COLOR, RUN_COLORS, SCATTER_MARKERS
from gabbro.plotting.writeup.summary import add_writeup_header

SUITES = ("ttbar", "sm_mixture", "gghbb", "minbias")
SELECTIONS = ("full500", "puppi128")
PID_LABELS = (r"$h^0$", r"$\gamma$", r"$h^-$", r"$h^+$", r"$e^-$", r"$e^+$", r"$\mu^-$", r"$\mu^+$")
SUITE_LABELS = {
    "ttbar": r"$t\bar{t}$, balanced decay modes (10k)",
    "sm_mixture": "balanced SM mixture (10k)",
    "gghbb": r"$gg\to H\to b\bar{b}$ (10k)",
    "minbias": "minimum bias (10k)",
}
SUITE_NAMES = {
    "ttbar": r"$t\bar{t}$, balanced decay modes",
    "sm_mixture": "balanced SM mixture",
    "gghbb": r"$gg\to H\to b\bar{b}$",
    "minbias": "minimum bias",
}
SELECTION_LABELS = {
    "full500": r"Unfiltered leading-$p_T$ particles  |  $N_{\max}=500$",
    "puppi128": r"Pileup-subtracted  |  $w_{\mathrm{PUPPI}}>0.05$  |  $N_{\max}=128$",
}
MODEL_TRAINING_LABEL = r"5M-event $t\bar{t}$ / balanced-SM models"


def load_registry(path: Path):
    rows = yaml.safe_load(path.read_text())["models"]
    if len(rows) != 8:
        raise ValueError("Expected eight comparison models")
    for index, row in enumerate(rows):
        row["color"] = RUN_COLORS[index]
        row["linestyle"] = ("-", "--", "-.", ":")[index % 4]
        row["marker"] = SCATTER_MARKERS[index]
    return rows


def save(figure, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Render the raster copy first.  With constrained layout, the PDF backend's
    # tight-bbox pass can otherwise mutate the layout before the PNG render and
    # clip the upper line of a two-line figure-level identity block.
    for suffix, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        figure.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(figure)


def density(counts, bins):
    counts = np.asarray(counts, dtype=float)
    return counts / max(np.sum(counts * np.diff(bins)), 1e-12)


def safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def model_cache(cache_dir, suite, model):
    path = cache_dir / suite / f"{model['id']}.npz"
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise FileNotFoundError(path)
    return np.load(path), json.loads(path.with_suffix(".json").read_text())


def add_campaign_header(figure, axes, *, test_text, selection_text):
    """Add the standard title-free identity block below an external legend."""
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.82))
    figure.canvas.draw()
    right = axes[-1].get_position().x1
    figure.text(
        right,
        0.895,
        f"Train: {MODEL_TRAINING_LABEL}  |  Test: {test_text}",
        ha="right",
        va="top",
        fontsize=12.5,
    )
    figure.text(
        right,
        0.857,
        selection_text,
        ha="right",
        va="top",
        fontsize=12.5,
    )


def external_legend(figure, axes, models):
    handles = [axes[0].plot([], [], color=m["color"], linestyle=m["linestyle"], linewidth=2.0, label=m["label"])[0] for m in models]
    figure.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.5,
        handlelength=2.0,
    )


def plot_particle(cache_dir, suite, models):
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.3), constrained_layout=True)
    specs = (("pT", r"$p_T^{dec.}-p_T^{orig.}$ [GeV]"), ("Eta", r"$\eta^{dec.}-\eta^{orig.}$"), ("Phi", r"wrapped $\phi^{dec.}-\phi^{orig.}$"))
    for model in models:
        cache, _ = model_cache(cache_dir, suite, model)
        for axis, (feature, xlabel) in zip(axes, specs, strict=True):
            bins = cache[f"particle_{feature}_diff_bins"]
            axis.stairs(cache[f"particle_{feature}_diff_counts"], bins, color=model["color"], linestyle=model["linestyle"], linewidth=1.8)
            axis.set(xlabel=xlabel, yscale="log")
            axis.axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("Density")
    external_legend(figure, axes, models)
    add_campaign_header(
        figure,
        axes,
        test_text=SUITE_LABELS[suite],
        selection_text=(
            r"Inputs compared: unfiltered $N_{\max}=500$ and "
            r"$w_{\mathrm{PUPPI}}>0.05$, $N_{\max}=128$"
        ),
    )
    return figure


def plot_jets(cache_dir, suite, models):
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.3), constrained_layout=True)
    specs = (("delta_pt", (-1, 1), r"$(p_T^{dec.}-p_T^{orig.})/p_T^{orig.}$"), ("delta_eta", (-1, 1), r"$\eta^{dec.}-\eta^{orig.}$"), ("delta_phi", (-1, 1), r"wrapped $\phi^{dec.}-\phi^{orig.}$"))
    for model in models:
        cache, _ = model_cache(cache_dir, suite, model)
        for axis, (key, limits, xlabel) in zip(axes, specs, strict=True):
            edges = np.linspace(*limits, 81)
            counts, _ = np.histogram(cache[f"jet_{key}"], bins=edges, density=True)
            axis.stairs(counts, edges, color=model["color"], linestyle=model["linestyle"], linewidth=1.8)
            axis.set(xlabel=xlabel, xlim=limits, yscale="log")
            axis.axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("Density")
    external_legend(figure, axes, models)
    add_campaign_header(
        figure,
        axes,
        test_text=SUITE_LABELS[suite],
        selection_text=(
            r"AK4 anti-$k_t$, $R=0.4$, $p_T\geq30$ GeV  |  "
            r"Hungarian pairing, no $\Delta R$ rejection"
        ),
    )
    return figure


def plot_pid(cache_dir, suite, model):
    cache, _ = model_cache(cache_dir, suite, model)
    hep.style.use("CMS")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.1), constrained_layout=True)
    specs = (("delta_eta", (-2, 2), r"$\eta^{dec.}-\eta^{orig.}$"), ("delta_phi", (-1, 1), r"wrapped $\phi^{dec.}-\phi^{orig.}$"), ("delta_log_pt", (-1.5, 1.5), r"$\log(p_T^{dec.}/p_T^{orig.})$"))
    pid = cache["pid_original"].astype(int)
    for axis, (key, limits, xlabel) in zip(axes, specs, strict=True):
        edges = np.linspace(*limits, 81)
        for value, label in enumerate(PID_LABELS):
            axis.hist(cache[f"pid_{key}"][pid == value], bins=edges, density=True, histtype="step", linewidth=1.5, color=RUN_COLORS[value], label=label)
        axis.set(xlabel=xlabel, xlim=limits, yscale="log")
        axis.axvline(0, color="0.4", linewidth=0.8)
    axes[0].set_ylabel("Density")
    axes[-1].legend(ncol=2, frameon=False, fontsize=12)
    add_writeup_header(
        figure,
        axes,
        training_text=(
            r"balanced SM mixture (5M), PID-weighted VQ-STE, "
            f"{model['codes']} codes"
        ),
        test_text=SUITE_LABELS[suite],
        selection_text=SELECTION_LABELS[model["selection"]],
    )
    return figure


def load_pid_confusion(model, suite):
    source = (
        Path(model["pid_confusion_dir"])
        / suite
        / "artifacts"
        / "pid_confusion_matrix_test_suite_0.json"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text())
    counts = np.asarray(payload["counts"], dtype=np.int64)
    expected_shape = (len(PID_LABELS), len(PID_LABELS))
    if counts.shape != expected_shape or np.any(counts < 0):
        raise ValueError(f"Invalid PID confusion counts in {source}: {counts.shape}")
    normalized = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts, dtype=float),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )
    return source, counts, normalized, payload


def plot_pid_confusions(models, suite, selection):
    selected = [
        model
        for model in models
        if model.get("pid_confusion_dir")
        and model["selection"] == selection
        and model["codes"] == 4096
        and model["quantizer"] == "VQ-STE"
    ]
    if not selected:
        raise ValueError("No PID confusion sources are configured")
    selected.sort(
        key=lambda model: (
            2
            if model.get("pid_weighted")
            else 0
            if "_tt_" in model["id"]
            else 1
        )
    )
    hep.style.use("CMS")
    figure, axes = plt.subplots(
        1,
        len(selected),
        figsize=(6.1 * len(selected), 6.7),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    values = {}
    source_paths = {}
    event_counts = []
    image = None
    for index, (axis, model) in enumerate(zip(axes, selected, strict=True)):
        source, counts, normalized, payload = load_pid_confusion(model, suite)
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(len(PID_LABELS)):
            for column in range(len(PID_LABELS)):
                value = normalized[row, column]
                if value >= 0.001 or row == column:
                    axis.text(
                        column,
                        row,
                        f"{100 * value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if value > 0.55 else "black",
                    )
        panel_title = (
            "PID-weighted loss"
            if model.get("pid_weighted")
            else "tt training"
            if "_tt_" in model["id"]
            else "SM training"
        )
        axis.set(
            title=panel_title,
            xlabel="Decoded PID",
            ylabel="Original PID" if index == 0 else "",
            xticks=np.arange(len(PID_LABELS)),
            yticks=np.arange(len(PID_LABELS)),
            xticklabels=PID_LABELS,
            yticklabels=PID_LABELS if index == 0 else [],
        )
        axis.tick_params(axis="both", labelsize=12)
        values[f"{model['id']}_counts"] = counts
        values[f"{model['id']}_row_normalized"] = normalized
        source_paths[model["id"]] = str(source)
        if payload.get("events") is not None:
            event_counts.append(int(payload["events"]))
        elif model["selection"] == "full500" and counts.sum() % 500 == 0:
            event_counts.append(int(counts.sum() // 500))
    if image is None:
        raise RuntimeError("PID confusion rendering produced no panels")
    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02)
    colorbar.set_label("Fraction of original PID class", fontsize=14)
    colorbar.ax.tick_params(labelsize=11)
    if event_counts and len(set(event_counts)) == 1:
        event_count = event_counts[0]
        events_text = (
            f"{event_count / 1e6:.2f}M events"
            if event_count >= 1_000_000
            else f"{event_count / 1e3:g}k events"
            if event_count >= 1_000
            else f"{event_count} events"
        )
    else:
        events_text = "full test sample"
    add_writeup_header(
        figure,
        axes,
        training_text=(
            r"5M-event $t\bar{t}$ / balanced-SM models, VQ-STE, 4096 codes"
        ),
        test_text=f"{SUITE_NAMES[suite]} ({events_text})",
        selection_text=(
            SELECTION_LABELS[selection]
            + "  |  rows normalized independently"
        ),
    )
    metadata = {
        "suite": suite,
        "events": event_counts[0] if event_counts and len(set(event_counts)) == 1 else None,
        "normalization": "rows sum to one; original PID on rows, decoded PID on columns",
        "sources": source_paths,
    }
    return figure, values, metadata


def select_downstream_models(models, selection, *, paper):
    """Select one particle representation and optionally omit diagnostic losses."""
    return [
        model
        for model in models
        if model["selection"] == selection
        and (not paper or not model.get("pid_weighted"))
    ]


def plot_downstream(cache_dir, selection, models, kind):
    suite = "gghbb" if kind == "higgs" else "tau32"
    hep.style.use("CMS")
    figure = plt.figure(figsize=(21.0, 7.2), constrained_layout=True)
    outer = figure.add_gridspec(1, 3)
    distribution_axes = []
    ratio_axes = []
    for column in range(2):
        grid = outer[0, column].subgridspec(
            2, 1, height_ratios=(3.2, 1.0), hspace=0.04
        )
        distribution_axes.append(figure.add_subplot(grid[0]))
        ratio_axes.append(figure.add_subplot(grid[1], sharex=distribution_axes[-1]))
    residual_axis = figure.add_subplot(outer[0, 2])
    payloads = {
        model["id"]: model_cache(cache_dir, suite, model)[0] for model in models
    }
    first = payloads[models[0]["id"]]
    if kind == "higgs":
        panels = (("higgs_n_jets_original", "higgs_n_jets_decoded", np.arange(-0.5, 30.5, 1), "Number of AK4 jets"), ("higgs_mass_original", "higgs_mass_decoded", np.arange(40, 405, 5), "Higgs candidate (dijet) mass [GeV]"))
        residual_keys = ("higgs_mass_original", "higgs_mass_decoded")
        residual_limit = 400.0 if selection == "full500" else 200.0
        residual_xlabel = (
            r"$m_{H,\mathrm{cand.}}^{\mathrm{dec.}}"
            r"-m_{H,\mathrm{cand.}}^{\mathrm{orig.}}$ [GeV]"
        )
    else:
        panels = (("original_count", "decoded_count", np.arange(-0.5, 10.5, 1), r"Number of AK8 jets ($p_T\geq450$ GeV)"), ("original_tau32", "decoded_tau32", np.linspace(0, 1.2, 51), r"AK8 jet $\tau_{32}$"))
        residual_keys = ("original_tau32", "decoded_tau32")
        residual_limit = 0.8
        residual_xlabel = r"Decoded $-$ original AK8 jet $\tau_{32}$"
    # Event-level counts and Higgs candidates are common inputs and must agree
    # exactly.  Tau32 is evaluated only for successfully matched jet pairs, so
    # its original population is legitimately model-dependent.
    shared_original_keys = {panels[0][0]}
    if kind == "higgs":
        shared_original_keys.add(panels[1][0])
    for model in models[1:]:
        payload = payloads[model["id"]]
        for key in shared_original_keys:
            if not np.array_equal(first[key], payload[key], equal_nan=True):
                raise ValueError(
                    f"Original downstream values differ for {model['id']} ({suite}, {key}); "
                    "a shared tokenized/original denominator would be invalid"
                )
    for column, (axis, ratio_axis, panel) in enumerate(
        zip(distribution_axes, ratio_axes, panels, strict=True)
    ):
        orig_key, reco_key, bins, xlabel = panel
        original_densities = []
        edges = bins
        for model in models:
            original = payloads[model["id"]][orig_key]
            finite_original = original[np.isfinite(original)]
            original_counts, edges = np.histogram(finite_original, bins=bins)
            original_densities.append(density(original_counts, edges))
        original_density = np.mean(original_densities, axis=0)
        originals_are_shared = all(
            np.array_equal(original_densities[0], values, equal_nan=True)
            for values in original_densities[1:]
        )
        axis.stairs(
            original_density,
            edges,
            fill=True,
            alpha=0.32,
            color=ORIGINAL_COLOR,
            label="Original" if originals_are_shared else "Original (matched samples)",
        )
        if not originals_are_shared:
            axis.fill_between(
                edges[:-1],
                np.min(original_densities, axis=0),
                np.max(original_densities, axis=0),
                step="post",
                color=ORIGINAL_COLOR,
                alpha=0.18,
                linewidth=0,
            )
        for model, paired_original_density in zip(
            models, original_densities, strict=True
        ):
            cache = payloads[model["id"]]
            values = cache[reco_key]
            decoded_counts, _ = np.histogram(values[np.isfinite(values)], bins=edges)
            decoded_density = density(decoded_counts, edges)
            axis.stairs(
                decoded_density,
                edges,
                linewidth=1.8,
                color=model["color"],
                linestyle=model["linestyle"],
                label=model["label"],
            )
            ratio_axis.stairs(
                safe_ratio(decoded_density, paired_original_density),
                edges,
                linewidth=1.5,
                color=model["color"],
                linestyle=model["linestyle"],
            )
        axis.set(ylabel="Normalized density")
        axis.tick_params(labelbottom=False)
        ratio_axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        ratio_axis.set(
            xlabel=xlabel,
            ylabel="Tokenized/original" if column == 0 else "",
            ylim=(0.5, 1.5),
        )
    residual_bins = np.linspace(-residual_limit, residual_limit, 81)
    for model in models:
        cache = payloads[model["id"]]
        original = cache[residual_keys[0]]
        decoded = cache[residual_keys[1]]
        paired = np.isfinite(original) & np.isfinite(decoded)
        residual_axis.hist(
            decoded[paired] - original[paired],
            bins=residual_bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=model["color"],
            linestyle=model["linestyle"],
            label=model["label"],
        )
    residual_axis.axvline(0, color="0.35", linestyle="--", linewidth=1.0)
    residual_axis.set(
        xlabel=residual_xlabel,
        ylabel="Normalized density",
        xlim=(-residual_limit, residual_limit),
    )
    distribution_axes[0].set_yscale("log")
    handles, labels = distribution_axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
        columnspacing=1.5,
        handlelength=2.0,
    )
    if kind == "higgs":
        test_text = r"$gg\to H\to b\bar{b}$ (10k)"
        analysis_text = (
            SELECTION_LABELS[selection]
            + r"  |  AK4 anti-$k_t$, $R=0.4$, $p_T\geq30$ GeV"
        )
    else:
        test_text = r"all-hadronic $t\bar{t}$ (10k)"
        analysis_text = (
            SELECTION_LABELS[selection]
            + r"  |  AK8 original $p_T\geq450$ GeV  |  "
            r"Hungarian $\Delta R<0.4$"
        )
    add_campaign_header(
        figure,
        [*distribution_axes, residual_axis],
        test_text=test_text,
        selection_text=analysis_text,
    )
    figure.canvas.draw()
    figure.set_layout_engine(None)
    return figure


def plot_z_mumu(cache_dir, model):
    cache_root = cache_dir / "z_mumu" / model["id"]
    histograms = np.load(cache_root / "z_mumu_mass_histograms.npz")
    metrics = json.loads((cache_root / "z_mumu_mass_metrics.json").read_text())
    summary = metrics["z_mumu"]
    original_fit = summary["original_fit"]
    decoded_fit = summary["decoded_fit"]

    hep.style.use("CMS")
    figure = plt.figure(figsize=(15.5, 7.2), constrained_layout=True)
    outer = figure.add_gridspec(1, 2)
    mass_grid = outer[0, 0].subgridspec(
        2, 1, height_ratios=(3.2, 1.0), hspace=0.04
    )
    axis = figure.add_subplot(mass_grid[0])
    ratio_axis = figure.add_subplot(mass_grid[1], sharex=axis)
    residual_axis = figure.add_subplot(outer[0, 1])
    bins = histograms["bins"]
    original_counts, edges = np.histogram(histograms["original"], bins=bins)
    decoded_counts, _ = np.histogram(histograms["decoded"], bins=edges)
    original_density = density(original_counts, edges)
    decoded_density = density(decoded_counts, edges)
    axis.stairs(
        original_density,
        edges,
        fill=True,
        alpha=0.32,
        color=ORIGINAL_COLOR,
        label=rf"Original ($\mu={original_fit['mean']:.1f}$, $\sigma={original_fit['sigma']:.1f}$ GeV)",
    )
    axis.stairs(
        decoded_density,
        edges,
        linewidth=2.0,
        color=model["color"],
        label=rf"Decoded ($\mu={decoded_fit['mean']:.1f}$, $\sigma={decoded_fit['sigma']:.1f}$ GeV)",
    )
    axis.axvline(91.1876, color="black", linestyle="--", linewidth=1.2, label=r"$m_Z$")
    axis.set(
        ylabel="Normalized density",
        xlim=(60, 120),
    )
    axis.tick_params(labelbottom=False)
    axis.legend(frameon=False, fontsize=11)
    ratio_axis.stairs(
        safe_ratio(decoded_density, original_density),
        edges,
        linewidth=1.6,
        color=model["color"],
    )
    ratio_axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    ratio_axis.set(
        xlabel=r"$Z\to\mu^+\mu^-$ candidate mass [GeV]",
        ylabel="Tokenized/original",
        xlim=(60, 120),
        ylim=(0.5, 1.5),
    )
    mass_residual = histograms["decoded"] - histograms["original"]
    residual_axis.hist(
        mass_residual[np.isfinite(mass_residual)],
        bins=np.linspace(-100.0, 100.0, 81),
        density=True,
        histtype="stepfilled",
        alpha=0.22,
        color=model["color"],
    )
    residual_axis.hist(
        mass_residual[np.isfinite(mass_residual)],
        bins=np.linspace(-100.0, 100.0, 81),
        density=True,
        histtype="step",
        linewidth=2.0,
        color=model["color"],
    )
    residual_axis.axvline(0, color="0.35", linestyle="--", linewidth=1.0)
    residual_axis.set(
        xlabel=r"$m_{\mu\mu}^{\mathrm{dec.}}-m_{\mu\mu}^{\mathrm{orig.}}$ [GeV]",
        ylabel="Normalized density",
        xlim=(-100, 100),
    )
    add_writeup_header(
        figure,
        [axis, residual_axis],
        training_text=(
            r"balanced SM mixture (5M), "
            + ("PID-weighted " if model.get("pid_weighted") else "")
            + f"VQ-STE, {model['codes']} codes"
        ),
        test_text=r"leptonic $ZZ$ (10k)",
        selection_text=(
            SELECTION_LABELS["puppi128"]
            + r"  |  independent leading-$p_T$ $\mu^-$ and $\mu^+$"
        ),
    )
    figure.canvas.draw()
    figure.set_layout_engine(None)
    return figure


def plot_z_mumu_muons(cache_dir, model, subset):
    cache_root = cache_dir / "z_mumu" / model["id"]
    with np.load(cache_root / "z_mumu_muon_features.npz") as features:
        original = features[f"{subset}_original"].copy()
        decoded = features[f"{subset}_decoded"].copy()
    if original.shape != decoded.shape or original.ndim != 2 or original.shape[1] != 3:
        raise ValueError(f"Invalid {subset} muon feature cache")

    delta_pt = decoded[:, 2] - original[:, 2]
    delta_eta = decoded[:, 0] - original[:, 0]
    delta_phi = np.remainder(decoded[:, 1] - original[:, 1] + np.pi, 2 * np.pi) - np.pi

    hep.style.use("CMS")
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.2), constrained_layout=True)
    top_specs = (
        (2, np.geomspace(1.5, 500.0, 61), r"Muon $p_T$ [GeV]", True),
        (0, np.linspace(-2.7, 2.7, 61), r"Muon $\eta$", False),
        (1, np.linspace(-np.pi, np.pi, 65), r"Muon $\phi$", False),
    )
    for column, (feature, bins, xlabel, log_x) in enumerate(top_specs):
        axis = axes[0, column]
        axis.hist(
            original[:, feature],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.32,
            color=ORIGINAL_COLOR,
            label="Original",
        )
        axis.hist(
            decoded[:, feature],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=model["color"],
            label="Decoded",
        )
        axis.set(xlabel=xlabel, ylabel="Normalized density")
        if log_x:
            axis.set_xscale("log")

    bottom_specs = (
        (
            delta_pt,
            np.linspace(-50.0, 50.0, 81),
            r"$p_T^{dec.}-p_T^{orig.}$ [GeV]",
        ),
        (delta_eta, np.linspace(-2.0, 2.0, 81), r"$\eta^{dec.}-\eta^{orig.}$"),
        (delta_phi, np.linspace(-0.5, 0.5, 81), r"wrapped $\phi^{dec.}-\phi^{orig.}$"),
    )
    for axis, (values, bins, xlabel) in zip(axes[1], bottom_specs, strict=True):
        axis.hist(
            values[np.isfinite(values)],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.22,
            color=model["color"],
        )
        axis.hist(
            values[np.isfinite(values)],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=model["color"],
        )
        axis.axvline(0, color="black", linestyle="--", linewidth=1.0)
        axis.set(xlabel=xlabel, ylabel="Normalized density")

    axes[0, 2].legend(frameon=False, fontsize=13)
    population = (
        f"all original-PID muons ({len(original):,})"
        if subset == "all"
        else f"common leading-$p_T$ $Z$ candidates ({len(original) // 2:,} events)"
    )
    add_writeup_header(
        figure,
        list(axes.flat),
        training_text=r"balanced SM mixture (5M), PID-weighted VQ-STE, 4096 codes",
        test_text=r"leptonic $ZZ$ (10k)",
        selection_text=SELECTION_LABELS["puppi128"] + "  |  " + population,
    )
    # Keep both identity lines safely inside the canvas when these large
    # constrained-layout figures are rendered sequentially by different
    # Matplotlib backends.
    figure.texts[-2].set_y(0.975)
    figure.texts[-1].set_y(0.942)
    figure.canvas.draw()
    figure.set_layout_engine(None)
    return figure


def plot_rate_distortion(cache_dir, canonical_manifest, models):
    records = json.loads(canonical_manifest.read_text())
    canonical = sorted((float(row["metrics/rate/marginal_bits_per_input_particle"]) / 40.0, float(row["plot_metrics/reco_mse_total"])) for row in records if row.get("plot_family") == "VQ STE" and row.get("metrics/rate/marginal_bits_per_input_particle") is not None and row.get("plot_metrics/reco_mse_total") is not None)
    hep.style.use("CMS")
    figure, axis = plt.subplots(figsize=(9.5, 6.8), constrained_layout=True)
    axis.plot([x for x, _ in canonical], [y for _, y in canonical], color="0.3", marker="s", markerfacecolor="white", linewidth=1.8, label=r"Full-500 $t\bar{t}$ VQ-STE (200k)")
    values = []
    for model in models:
        _, summary = model_cache(cache_dir, "ttbar", model)
        x = summary["metrics/rate/marginal_bits_per_input_particle"] / 40.0
        y = summary["metrics/mse_total"]
        axis.scatter(x, y, s=90, color=model["color"], marker=model["marker"], edgecolor="black", linewidth=0.6, label=f"{model['gimmick']} (5M)", zorder=3)
        values.append({"model": model["id"], "gimmick": model["gimmick"], "rate_over_40bit": x, "mse": y})
    axis.set(xlabel="Marginal compressed rate / 40-bit input candidate", ylabel="Reconstruction MSE", yscale="log")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=10, ncol=2)
    add_writeup_header(
        figure,
        [axis],
        training_text=r"5M-event models and $t\bar{t}$ reference (200k events)",
        test_text=SUITE_LABELS["ttbar"],
        selection_text=(
            r"Unfiltered $N_{\max}=500$ and pileup-subtracted $N_{\max}=128$  |  "
            r"rate normalized to the 40-bit input payload"
        ),
    )
    return figure, values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--fixed-comparison-dir", type=Path, required=True)
    parser.add_argument("--fixed-puppi128-comparison-dir", type=Path)
    parser.add_argument("--fixed-puppi128-sm-comparison-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    models = load_registry(args.registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = []
    for suite in SUITES:
        save(plot_particle(args.cache_dir, suite, models), args.output_dir / "residuals" / "particles" / suite)
        save(plot_jets(args.cache_dir, suite, models), args.output_dir / "residuals" / "ak4_jets" / suite)
        products.extend((f"residuals/particles/{suite}", f"residuals/ak4_jets/{suite}"))
        for model in (row for row in models if row.get("pid_weighted")):
            save(plot_pid(args.cache_dir, suite, model), args.output_dir / "pid_residuals" / model["selection"] / suite)
            products.append(f"pid_residuals/{model['selection']}/{suite}")
        for selection in SELECTIONS:
            figure, confusion_values, confusion_metadata = plot_pid_confusions(
                models, suite, selection
            )
            confusion_stem = args.output_dir / "pid_confusion" / selection / suite
            save(figure, confusion_stem)
            np.savez_compressed(
                confusion_stem.with_suffix(".npz"),
                **confusion_values,
            )
            confusion_stem.with_suffix(".json").write_text(
                json.dumps(confusion_metadata, indent=2) + "\n"
            )
            products.append(f"pid_confusion/{selection}/{suite}")
    for selection in SELECTIONS:
        selected = select_downstream_models(models, selection, paper=False)
        paper_selected = select_downstream_models(models, selection, paper=True)
        for kind in ("higgs", "tau32"):
            save(
                plot_downstream(args.cache_dir, selection, selected, kind),
                args.output_dir / "downstream" / "all_models" / kind / selection,
            )
            products.append(f"downstream/all_models/{kind}/{selection}")
            save(
                plot_downstream(args.cache_dir, selection, paper_selected, kind),
                args.output_dir / "downstream" / "paper" / kind / selection,
            )
            products.append(f"downstream/paper/{kind}/{selection}")
    z_model = next(
        model for model in models
        if model["id"] == "puppi128_sm_pidweighted_vq4096"
    )
    save(
        plot_z_mumu(args.cache_dir, z_model),
        args.output_dir / "downstream" / "all_models" / "z_mumu" / "puppi128",
    )
    products.append("downstream/all_models/z_mumu/puppi128")
    paper_z_model = next(
        model for model in models
        if model["id"] == "puppi128_sm_vq4096"
    )
    save(
        plot_z_mumu(args.cache_dir, paper_z_model),
        args.output_dir / "downstream" / "paper" / "z_mumu" / "puppi128",
    )
    products.append("downstream/paper/z_mumu/puppi128")
    for subset, stem in (
        ("all", "puppi128_muons_all"),
        ("z_candidates", "puppi128_muons_z_candidates"),
    ):
        save(
            plot_z_mumu_muons(args.cache_dir, z_model, subset),
            args.output_dir / "downstream" / "all_models" / "z_mumu" / stem,
        )
        products.append(f"downstream/all_models/z_mumu/{stem}")
    fixed_target = args.output_dir / "training_scale" / "matched_tt"
    fixed_target.mkdir(parents=True, exist_ok=True)
    for path in args.fixed_comparison_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, fixed_target / path.name)
    if args.fixed_puppi128_comparison_dir is not None:
        puppi_target = fixed_target / "puppi128"
        puppi_target.mkdir(parents=True, exist_ok=True)
        for path in args.fixed_puppi128_comparison_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, puppi_target / path.name)
    if args.fixed_puppi128_sm_comparison_dir is not None:
        sm_target = args.output_dir / "training_scale" / "matched_sm" / "puppi128"
        sm_target.mkdir(parents=True, exist_ok=True)
        for path in args.fixed_puppi128_sm_comparison_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, sm_target / path.name)
    figure, rate_values = plot_rate_distortion(args.cache_dir, args.canonical_manifest, models)
    save(figure, args.output_dir / "training_scale" / "all_eight_rate_distortion")
    (args.output_dir / "training_scale" / "all_eight_rate_distortion.json").write_text(json.dumps(rate_values, indent=2) + "\n")
    products.extend(("training_scale/matched_tt", "training_scale/all_eight_rate_distortion"))
    if args.fixed_puppi128_sm_comparison_dir is not None:
        products.append("training_scale/matched_sm/puppi128")
    status = {"status": "complete", "models": [{k: v for k, v in row.items() if k not in {"color", "linestyle", "marker"}} for row in models], "products": products, "cache_dir": str(args.cache_dir), "output_dir": str(args.output_dir)}
    (args.output_dir / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Joint 5M model comparison\n\n"
        "All plots are title-free PDF and 300-dpi PNG products. See `STATUS.json` "
        "for the complete model and product inventory. AK4 residual jets use "
        "unconstrained minimum-total-$\\Delta R$ Hungarian assignment. The "
        "`downstream/all_models/` figures retain every model; the paper-ready "
        "copies under `downstream/paper/` omit PID-weighted models. Both sets "
        "include tokenized/original ratio panels.\n"
    )
    print(f"Rendered {len(products)} plot groups in {args.output_dir}")


if __name__ == "__main__":
    main()
