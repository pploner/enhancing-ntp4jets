"""ORBIT-style plotting helpers for schema-neutral tokenization outputs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

import gabbro.plotting.utils as plot_utils

ORIGINAL_COLOR = plot_utils.DEFAULT_COLORS[0]
RECONSTRUCTED_COLOR = plot_utils.DEFAULT_COLORS[1]
RESIDUAL_COLOR = plot_utils.DEFAULT_COLORS[1]
TRUTH_REFERENCE_COLOR = "grey"
RUN_COLORS = tuple(plot_utils.DEFAULT_COLORS)
# Keep codebook-size-only legends readable: the generic hash fallback can map
# different numeric labels onto the same finite palette.  These entries cover
# the canonical VQ scans and are intentionally stable across all figures.
CODEBOOK_SIZE_COLORS = {
    label: RUN_COLORS[index]
    for index, label in enumerate(("128", "256", "512", "1024", "2048", "4096", "8192", "16384"))
}
CODEBOOK_FAMILY_COLORS = {
    "fsq": plot_utils.DEFAULT_COLORS[0],
    "vq_ste": plot_utils.DEFAULT_COLORS[1],
    "vq_rotation": plot_utils.DEFAULT_COLORS[2],
    "kmeans": plot_utils.DEFAULT_COLORS[3],
    "scalar_baseline": plot_utils.DEFAULT_COLORS[4],
    "continuous": plot_utils.DEFAULT_COLORS[5],
    # Preserve the identities used by the tt/ggHbb VQ-STE training-domain
    # comparisons for all equivalent display labels.
    "training_tt": plot_utils.DEFAULT_COLORS[3],
    "training_sm_mixture": plot_utils.DEFAULT_COLORS[7],
}
CODEBOOK_FAMILY_MARKERS = {
    "fsq": "o",
    "vq_ste": "s",
    "vq_rotation": "^",
    "kmeans": "D",
    "split_vq_fsq_alpha_128": "v",
    "split_vq_fsq_alpha_64": "P",
    "split_vq_fsq_alpha_32": "X",
    "split_fsq_alpha_128": "*",
    "split_fsq_alpha_64": "<",
    "split_fsq_alpha_32": ">",
    "training_tt": "D",
    "training_sm_mixture": "P",
}
MIN_CODEBOOK_MARKER_AREA = 55.0
MAX_CODEBOOK_MARKER_AREA = 210.0
PRESENTATION_CODEBOOK_MARKER_AREA_SCALE = 0.55
CODEBOOK_FAMILY_COLORS.update(
    {
        "split_vq_fsq_alpha_128": plot_utils.DEFAULT_COLORS[4],
        "split_vq_fsq_alpha_64": plot_utils.DEFAULT_COLORS[5],
        "split_vq_fsq_alpha_32": plot_utils.DEFAULT_COLORS[6],
        "split_fsq_alpha_128": plot_utils.DEFAULT_COLORS[7],
        "split_fsq_alpha_64": plot_utils.DEFAULT_COLORS[8],
        "split_fsq_alpha_32": plot_utils.DEFAULT_COLORS[9],
    }
)
CODEBOOK_FAMILY_LABELS = {
    "fsq": "FSQ",
    "vq_ste": "VQ STE",
    "vq_rotation": "VQ rotation",
    "kmeans": "FAISS k-means",
}
SCATTER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8")


def pid_residual_arrays(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Return same-position residuals in physical angular/log coordinates."""
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    if original.shape != reconstructed.shape or original.ndim != 2:
        raise ValueError("PID residual inputs must have matching [particles, features] shapes")
    feature_to_index = {str(name): index for index, name in enumerate(feature_names)}
    required = (
        "L1T_PUPPIPart_Eta",
        "L1T_PUPPIPart_Phi_cos",
        "L1T_PUPPIPart_Phi_sin",
        "L1T_PUPPIPart_PT",
    )
    missing = [name for name in required if name not in feature_to_index]
    if missing:
        raise ValueError(f"PID residual plots require features {missing}")

    eta = feature_to_index[required[0]]
    phi_cos = feature_to_index[required[1]]
    phi_sin = feature_to_index[required[2]]
    pt = feature_to_index[required[3]]
    original_phi = np.arctan2(original[:, phi_sin], original[:, phi_cos])
    reconstructed_phi = np.arctan2(
        reconstructed[:, phi_sin], reconstructed[:, phi_cos]
    )
    delta_phi = np.remainder(reconstructed_phi - original_phi + np.pi, 2 * np.pi) - np.pi
    residuals = {
        "delta_eta": 3.0 * (reconstructed[:, eta] - original[:, eta]),
        "delta_phi": delta_phi,
        "delta_log_pt": reconstructed[:, pt] - original[:, pt],
    }
    energy_name = "L1T_PUPPIPart_E"
    if energy_name in feature_to_index:
        energy = feature_to_index[energy_name]
        residuals["delta_log_energy"] = reconstructed[:, energy] - original[:, energy]
    return residuals


def pid_residual_distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize one PID-conditioned residual distribution."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "bias": None,
            "rmse": None,
            "median": None,
            "q16": None,
            "q84": None,
            "abs_q95": None,
        }
    return {
        "count": int(len(values)),
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "q16": float(np.quantile(values, 0.16)),
        "q84": float(np.quantile(values, 0.84)),
        "abs_q95": float(np.quantile(np.abs(values), 0.95)),
    }


def plot_pid_conditional_residuals(
    by_pid: dict[int, dict[str, np.ndarray]],
    class_names: list[str] | tuple[str, ...],
    bins: int = 80,
    title: str = "Same-token reconstruction residuals, conditioned on original PID",
):
    """Plot one residual panel per available kinematic coordinate."""
    plot_utils.set_mpl_style()
    specifications = [
        ("delta_eta", r"$\eta^\mathrm{reco}-\eta^\mathrm{orig}$", (-2.0, 2.0)),
        (
            "delta_phi",
            r"wrapped $\phi^\mathrm{reco}-\phi^\mathrm{orig}$",
            (-1.0, 1.0),
        ),
        (
            "delta_log_pt",
            r"$\log(p_T^\mathrm{reco}/p_T^\mathrm{orig})$",
            (-1.5, 1.5),
        ),
    ]
    if any("delta_log_energy" in values for values in by_pid.values()):
        specifications.append(
            (
                "delta_log_energy",
                r"$\log(E^\mathrm{reco}/E^\mathrm{orig})$",
                (-1.5, 1.5),
            )
        )
    figure, axes = plt.subplots(
        1,
        len(specifications),
        figsize=(6 * len(specifications), 5.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (key, label, value_range) in zip(axes, specifications):
        edges = np.linspace(*value_range, bins + 1)
        for pid, class_name in enumerate(class_names):
            values = np.asarray(by_pid.get(pid, {}).get(key, []))
            if not len(values):
                continue
            axis.hist(
                values,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=RUN_COLORS[pid % len(RUN_COLORS)],
                label=f"{class_name} ({len(values):,})",
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set(xlabel=label, ylabel="Density", xlim=value_range)
    axes[-1].legend(fontsize=7, frameon=False)
    figure.suptitle(title)
    return figure


def _multirun_family_key(series_name: str) -> str | None:
    """Return the canonical style key for a known multirun family."""
    name = str(series_name).strip()
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    normalized = (
        normalized.replace("μ", "mu")
        .replace("α", "alpha")
        .replace("+", "plus")
        .replace("=", "_")
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    family_aliases = {
        "fsq": "fsq",
        "fsq_mu_only": "fsq",
        "vq_ste": "vq_ste",
        "vq_rotation": "vq_rotation",
        "faiss_k_means": "kmeans",
        "faiss_kmeans": "kmeans",
        "kmeans": "kmeans",
        "scalar_baseline": "scalar_baseline",
        "dumb_quantization": "scalar_baseline",
        "continuous": "continuous",
        "continuous_autoencoder": "continuous",
        "tt_only": "training_tt",
        "trained_on_tt": "training_tt",
        "tt_trained_vq_ste": "training_tt",
        "sm_mixture": "training_sm_mixture",
        "trained_on_sm_mixture": "training_sm_mixture",
        "mixture_trained_vq_ste": "training_sm_mixture",
        "vq_mu_plus_fsq_alpha_128": "split_vq_fsq_alpha_128",
        "vq_mu_plus_fsq_alpha_64": "split_vq_fsq_alpha_64",
        "vq_mu_plus_fsq_alpha_32": "split_vq_fsq_alpha_32",
        "fsq_mu_plus_alpha_128": "split_fsq_alpha_128",
        "fsq_mu_plus_alpha_64": "split_fsq_alpha_64",
        "fsq_mu_plus_alpha_32": "split_fsq_alpha_32",
    }
    return family_aliases.get(normalized)


def multirun_color(series_name: str) -> str:
    """Return a stable color for a multirun series, independent of input order."""
    name = str(series_name).strip()
    if name in CODEBOOK_SIZE_COLORS:
        return CODEBOOK_SIZE_COLORS[name]
    family = _multirun_family_key(name)
    if family is not None:
        return CODEBOOK_FAMILY_COLORS[family]
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return RUN_COLORS[int.from_bytes(digest[:8], "big") % len(RUN_COLORS)]


def multirun_marker(series_name: str) -> str:
    """Return a stable marker for a multirun family, independent of input order."""
    name = str(series_name).strip()
    family = _multirun_family_key(name)
    if family in CODEBOOK_FAMILY_MARKERS:
        return CODEBOOK_FAMILY_MARKERS[family]
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return SCATTER_MARKERS[int.from_bytes(digest[8:16], "big") % len(SCATTER_MARKERS)]


def _phaedra_architecture_rank(series_name: str) -> int | None:
    """Return the desired vertical order for PHAEDRA split architectures."""
    normalized = (
        str(series_name)
        .strip()
        .lower()
        .replace("μ", "mu")
        .replace("α", "alpha")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "plus")
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    if normalized.startswith("vq_mu_plus_fsq_alpha"):
        return 0
    if normalized.startswith("fsq_mu_plus_alpha"):
        return 1
    return None


def multirun_display_label(series_name: str) -> str:
    """Return the presentation label for a multirun model family."""
    label = str(series_name)
    if _phaedra_architecture_rank(label) is not None and not label.startswith("(PHAEDRA)"):
        return f"(PHAEDRA) {label}"
    return label


def multirun_legend_sort_key(series_name: str) -> tuple[int, int, str]:
    """Put the two PHAEDRA architectures first and directly above one another."""
    rank = _phaedra_architecture_rank(series_name)
    return (0, rank, str(series_name)) if rank is not None else (1, 0, str(series_name))


HISTOGRAM_LINEWIDTH = 2
HISTOGRAM_FILL_ALPHA = 0.35
TRUTH_REFERENCE_FILL_ALPHA = 0.35
REFERENCE_LINE_COLOR = "black"
REFERENCE_LINE_STYLE = "--"
REFERENCE_LINE_ALPHA = 0.5
RATIO_YLIM = (0.5, 1.5)
FEATURE_FIGSIZE = (21, 8)
ENERGY_FIGSIZE = (16, 5)
RESOLUTION_FIGSIZE = (8, 6)
SUBSTRUCTURE_SIMPLE_FIGSIZE = (18, 8)
MONEY_TRIPLET_FIGSIZE = (21, 6)
RATIO_HEIGHT_RATIOS = (3, 1)
RATIO_HSPACE = 0.08
RATIO_WSPACE = 0.25
ATTENTION_DELTA_FIGSIZE = (6, 5)
ATTENTION_MAP_FIGSIZE = (5, 4)
SCATTER_FIGSIZE = (8, 6)
MULTIRUN_LEGEND_FONTSIZE = 8
PRESENTATION_LEGEND_FONTSIZE = 12
PRESENTATION_LEGEND_MARKERSCALE = 1.35
TRANSFORMED_FEATURE_RANGES = {
    "L1T_PUPPIPart_Eta": (-1.5, 1.5),
    "L1T_PUPPIPart_Phi_cos": (-1.0, 1.0),
    "L1T_PUPPIPart_Phi_sin": (-1.0, 1.0),
    "L1T_PUPPIPart_PT": (-2.0, 5.0),
}
TRANSFORMED_RESIDUAL_RANGE = (-0.5, 0.5)
PHYSICAL_FEATURE_RANGES = {
    "pT": (0.0, 2_000.0),
    "Eta": (-4.5, 4.5),
    "Phi": (-math.pi, math.pi),
}
PHYSICAL_RESIDUAL_RANGES = {
    "pT": (-50.0, 50.0),
    "Eta": (-2.0, 2.0),
    "Phi": (-1.0, 1.0),
}
MINBIAS_PHYSICAL_FEATURE_RANGES = {
    **PHYSICAL_FEATURE_RANGES,
    "pT": (0.0, 200.0),
}
ENERGY_RANGE = (0.0, 2_500.0)
ENERGY_RESIDUAL_RANGE = (-50.0, 50.0)
MISSING_ET_RANGE = (0.0, 1_000.0)
JET_MASS_RANGE = (0.0, 1_800.0)
JET_MASS_RESIDUAL_RANGE = (-50.0, 50.0)
TAU32_RESIDUAL_RANGE = (-0.4, 0.4)
KINEMATIC_LABELS = {
    "pT": r"$p_T$",
    "Eta": r"$\eta$",
    "Phi": r"$\phi$",
}
KINEMATIC_RESIDUAL_LABELS = {
    "pT": r"$p_T^\mathrm{reco} - p_T^\mathrm{orig}$",
    "Eta": r"$\eta^\mathrm{reco} - \eta^\mathrm{orig}$",
    "Phi": r"$\phi^\mathrm{reco} - \phi^\mathrm{orig}$",
}

PLOT_TITLES_ENABLED = True


def set_plot_titles_enabled(enabled: bool) -> None:
    global PLOT_TITLES_ENABLED
    PLOT_TITLES_ENABLED = enabled


def _set_title(ax, title, **kwargs) -> None:
    if PLOT_TITLES_ENABLED:
        ax.set_title(title, **kwargs)


def _set_suptitle(fig, title, **kwargs) -> None:
    if PLOT_TITLES_ENABLED:
        fig.suptitle(title, **kwargs)


def _feature_label(feature_name: str) -> str:
    physical_labels = {
        "physical_pT": r"$p_T$ [GeV]",
        "physical_Eta": r"$\eta$",
        "physical_Phi": r"$\phi$",
    }
    if feature_name in physical_labels:
        return physical_labels[feature_name]
    return plot_utils.DEFAULT_LABELS.get(feature_name, feature_name)


def _clean_feature_name(feature_name: str) -> str:
    return feature_name.replace("/", "_").replace(" ", "_")


def _masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return array[mask.astype(bool)]


def _finite_pair(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(original) & np.isfinite(reconstructed)
    return original[finite], reconstructed[finite]


def _linear_bins(value_range: tuple[float, float], n_bins: int) -> np.ndarray:
    return np.linspace(value_range[0], value_range[1], n_bins + 1)


def _clip_to_hist_range(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    return np.clip(values, bins[0], bins[-1])


def _density_hist_with_overflow(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(_clip_to_hist_range(values, bins), bins=bins, density=True)
    return np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)


def _counts_hist_with_overflow(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(_clip_to_hist_range(values, bins), bins=bins)
    return counts


def _angular_difference(reconstructed: np.ndarray, original: np.ndarray) -> np.ndarray:
    return np.remainder(reconstructed - original + np.pi, 2 * np.pi) - np.pi


def event_marker_areas(pt: np.ndarray) -> np.ndarray:
    """Return bounded, logarithmically scaled event-display marker areas."""
    log_pt = np.log10(np.clip(np.asarray(pt, dtype=float), 0.5, 500.0))
    fraction = (log_pt - np.log10(0.5)) / (np.log10(500.0) - np.log10(0.5))
    return 18.0 + 220.0 * np.clip(fraction, 0.0, 1.0)


def plot_particle_count_histograms(
    counts_by_class: dict[str, np.ndarray],
):
    """Plot combined and class-wise model-input particle multiplicities."""
    counts_by_class = {
        str(name): np.asarray(values, dtype=np.int64)
        for name, values in counts_by_class.items()
        if len(values)
    }
    if not counts_by_class:
        raise ValueError("counts_by_class must contain at least one event")

    plot_utils.set_mpl_style()
    combined = np.concatenate(list(counts_by_class.values()))
    maximum = int(np.max(combined)) if len(combined) else 0
    bins = np.arange(maximum + 2, dtype=np.float64) - 0.5
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].hist(
        combined,
        bins=bins,
        histtype="step",
        linewidth=HISTOGRAM_LINEWIDTH,
        color=ORIGINAL_COLOR,
        label=f"All classes (n={len(combined):,})",
    )
    axes[0].set_title("Joint training sample")
    axes[0].legend(prop={"size": 9})

    for index, (class_name, values) in enumerate(counts_by_class.items()):
        axes[1].hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=HISTOGRAM_LINEWIDTH,
            color=RUN_COLORS[index % len(RUN_COLORS)],
            label=f"{class_name} (n={len(values):,})",
        )
    axes[1].set_title("Training sample by class")
    axes[1].legend(prop={"size": 9})

    for axis in axes:
        axis.set_xlabel("Input particles per event")
        axis.set_ylabel("Events")
        axis.grid(alpha=0.25)
    return figure


def reconstruction_loss_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    reconstruction_loss: str,
) -> dict[str, float]:
    """Compute the model's masked reconstruction reductions on NumPy arrays."""
    if reconstruction_loss not in {"l1", "l2"}:
        raise ValueError(f"Unknown reconstruction_loss={reconstruction_loss!r}")
    valid_particles = np.clip(np.sum(mask), a_min=1, a_max=None)
    valid_values = valid_particles * original.shape[-1]
    delta = (reconstructed - original) * mask[..., None]
    l2 = float(np.sum(delta**2) / valid_particles)
    l1 = float(np.sum(np.abs(delta)) / valid_particles)
    l2_per_value = float(np.sum(delta**2) / valid_values)
    l1_per_value = float(np.sum(np.abs(delta)) / valid_values)
    return {
        "loss_reco": l1 if reconstruction_loss == "l1" else l2,
        "loss_reco_l1": l1,
        "loss_reco_l2": l2,
        "loss_reco_l1_per_value": l1_per_value,
        "loss_reco_l2_per_value": l2_per_value,
    }


@dataclass(frozen=True)
class CodeBranchSpec:
    """Mixed-radix metadata needed to unpack one quantizer branch."""

    name: str
    num_codes: int
    levels: tuple[int, ...] | None = None
    component_names: tuple[str, ...] | None = None


def _metric_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _empirical_entropy_bits(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return 0.0, 0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy), int(len(counts))


def code_entropy_metrics(
    code_idx: np.ndarray,
    code_mask: np.ndarray,
    input_mask: np.ndarray,
    num_codes: int | None,
    branch_specs: tuple[CodeBranchSpec, ...] = (),
) -> dict[str, float | int | None]:
    """Compute sparse marginal entropy and length-aware rate metrics."""
    codes = np.asarray(code_idx)
    latent_mask = np.asarray(code_mask, dtype=bool)
    particle_mask = np.asarray(input_mask, dtype=bool)
    if codes.ndim == latent_mask.ndim + 1 and codes.shape[-1] == 1:
        codes = codes[..., 0]
    if codes.shape != latent_mask.shape:
        raise ValueError(
            f"code_idx and code_mask must match, got {codes.shape} and {latent_mask.shape}"
        )
    if particle_mask.ndim != 2 or particle_mask.shape[0] != latent_mask.shape[0]:
        raise ValueError("input_mask must have one row per event")

    valid_codes = codes[latent_mask].astype(np.int64, copy=False)
    if np.any(valid_codes < 0):
        raise ValueError("Valid token IDs must be non-negative")
    if num_codes is not None and np.any(valid_codes >= int(num_codes)):
        raise ValueError("Valid token IDs must be smaller than num_codes")
    entropy, active_codes = _empirical_entropy_bits(valid_codes)
    n_events = int(latent_mask.shape[0])
    n_latent_tokens = int(latent_mask.sum())
    n_input_particles = int(particle_mask.sum())
    mean_latent_tokens = n_latent_tokens / max(n_events, 1)
    mean_input_particles = n_input_particles / max(n_events, 1)
    effective_token_ratio = n_latent_tokens / max(n_input_particles, 1)

    fixed_bits_per_token = None
    normalized_entropy = None
    fraction_of_fixed_width = None
    utilization = None
    total_codebook_size = None
    if num_codes is not None:
        total_codebook_size = int(num_codes)
        if total_codebook_size < 1:
            raise ValueError("num_codes must be positive")
        fixed_bits_per_token = int(total_codebook_size - 1).bit_length()
        utilization = active_codes / total_codebook_size
        if total_codebook_size > 1:
            normalized_entropy = entropy / math.log2(total_codebook_size)
        if fixed_bits_per_token > 0:
            fraction_of_fixed_width = entropy / fixed_bits_per_token

    metrics: dict[str, float | int | None] = {
        "metrics/active_codes_total": active_codes,
        "metrics/utilization_total": utilization,
        "metrics/total_codebook_size": total_codebook_size,
        "metrics/entropy/marginal_bits_per_token": entropy,
        "metrics/entropy/perplexity": float(2**entropy),
        "metrics/entropy/normalized_to_log2_codebook": normalized_entropy,
        "metrics/entropy/fixed_bits_per_token": fixed_bits_per_token,
        "metrics/entropy/fraction_of_fixed_width": fraction_of_fixed_width,
        "metrics/rate/mean_latent_tokens_per_event": mean_latent_tokens,
        "metrics/rate/mean_input_particles_per_event": mean_input_particles,
        "metrics/rate/effective_token_ratio": effective_token_ratio,
        "metrics/rate/marginal_bits_per_event": entropy * mean_latent_tokens,
        "metrics/rate/marginal_bits_per_input_particle": entropy * effective_token_ratio,
        "metrics/rate/fixed_bits_per_event": (
            None
            if fixed_bits_per_token is None
            else fixed_bits_per_token * mean_latent_tokens
        ),
        "metrics/rate/fixed_bits_per_input_particle": (
            None
            if fixed_bits_per_token is None
            else fixed_bits_per_token * effective_token_ratio
        ),
        "metrics/rate/sample_events": n_events,
        "metrics/rate/sample_latent_tokens": n_latent_tokens,
        "metrics/rate/sample_input_particles": n_input_particles,
    }

    if not branch_specs or valid_codes.size == 0:
        return metrics

    represented_codes = math.prod(spec.num_codes for spec in branch_specs)
    if num_codes is not None and represented_codes != int(num_codes):
        raise ValueError(
            "Branch codebook product must equal the combined codebook size, got "
            f"{represented_codes} and {num_codes}"
        )

    branch_entropy_sum = 0.0
    component_entropy_sum = 0.0
    packed_residual = valid_codes.copy()
    for branch_spec in branch_specs:
        if branch_spec.num_codes < 1:
            raise ValueError(f"Branch {branch_spec.name!r} must have at least one code")
        branch_codes = packed_residual % branch_spec.num_codes
        packed_residual //= branch_spec.num_codes
        branch_entropy, _ = _empirical_entropy_bits(branch_codes)
        branch_name = _metric_name(branch_spec.name)
        metrics[f"metrics/entropy/branch/{branch_name}/bits_per_token"] = branch_entropy
        branch_entropy_sum += branch_entropy

        if branch_spec.levels is None:
            metrics[
                f"metrics/entropy/component/{branch_name}/code/bits_per_token"
            ] = branch_entropy
            component_entropy_sum += branch_entropy
            continue

        if branch_spec.component_names is not None and (
            len(branch_spec.component_names) != len(branch_spec.levels)
        ):
            raise ValueError(
                f"Branch {branch_spec.name!r} component names and levels must match"
            )
        represented_branch_codes = math.prod(branch_spec.levels)
        if represented_branch_codes != branch_spec.num_codes:
            raise ValueError(
                f"Branch {branch_spec.name!r} levels represent "
                f"{represented_branch_codes} codes, not {branch_spec.num_codes}"
            )
        component_residual = branch_codes.copy()
        for component_index, level in enumerate(branch_spec.levels):
            if level < 1:
                raise ValueError(f"FSQ levels must be positive, got {level}")
            component_codes = component_residual % level
            component_residual //= level
            component_entropy, _ = _empirical_entropy_bits(component_codes)
            component_name = (
                branch_spec.component_names[component_index]
                if branch_spec.component_names is not None
                else f"dim_{component_index}"
            )
            component_name = _metric_name(component_name)
            metrics[
                f"metrics/entropy/component/{branch_name}/{component_name}/bits_per_token"
            ] = component_entropy
            component_entropy_sum += component_entropy
        if np.any(component_residual != 0):
            raise ValueError(
                f"Branch {branch_spec.name!r} codes exceed its FSQ morphology"
            )

    if np.any(packed_residual != 0):
        raise ValueError("Packed token IDs exceed the supplied branch morphology")

    metrics["metrics/entropy/branch_sum_marginals_bits_per_token"] = branch_entropy_sum
    metrics["metrics/entropy/branch_total_correlation_bits_per_token"] = max(
        branch_entropy_sum - entropy,
        0.0,
    )
    metrics["metrics/entropy/component_sum_marginals_bits_per_token"] = (
        component_entropy_sum
    )
    metrics["metrics/entropy/component_total_correlation_bits_per_token"] = max(
        component_entropy_sum - entropy,
        0.0,
    )
    return metrics


def plot_event_reconstruction_grid(
    event_groups: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    events_per_group: int = 3,
):
    """Plot matched original/reconstructed particle events in eta--phi space."""
    if events_per_group < 1:
        raise ValueError("events_per_group must be positive")
    if not event_groups:
        raise ValueError("event_groups must not be empty")

    plot_utils.set_mpl_style()
    fig, axes = plt.subplots(
        len(event_groups),
        events_per_group,
        figsize=(5.2 * events_per_group, 4.4 * len(event_groups)),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    cmap = plt.get_cmap("viridis")
    color_norm = Normalize(vmin=np.log10(0.5), vmax=np.log10(500.0), clip=True)

    for row, (group_name, original, reconstructed, mask) in enumerate(event_groups):
        for column in range(events_per_group):
            ax = axes[row, column]
            if column >= original.shape[0]:
                ax.text(0.5, 0.5, "No event available", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_axis_off()
                continue

            event_mask = mask[column].astype(bool)
            original_event = original[column, event_mask]
            reconstructed_event = reconstructed[column, event_mask]
            original_finite = np.all(np.isfinite(original_event), axis=1)
            reconstructed_finite = np.all(np.isfinite(reconstructed_event), axis=1)
            original_event = original_event[original_finite]
            reconstructed_event = reconstructed_event[reconstructed_finite]

            if original_event.size:
                ax.scatter(
                    original_event[:, 0],
                    original_event[:, 1],
                    s=event_marker_areas(original_event[:, 2]),
                    c=np.log10(np.clip(original_event[:, 2], 0.5, 500.0)),
                    cmap=cmap,
                    norm=color_norm,
                    marker="o",
                    alpha=0.65,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=2,
                )
            if reconstructed_event.size:
                ax.scatter(
                    reconstructed_event[:, 0],
                    reconstructed_event[:, 1],
                    s=event_marker_areas(reconstructed_event[:, 2]),
                    c=np.log10(np.clip(reconstructed_event[:, 2], 0.5, 500.0)),
                    cmap=cmap,
                    norm=color_norm,
                    marker="x",
                    alpha=0.9,
                    linewidths=1.0,
                    zorder=3,
                )

            ax.set_xlim(*PHYSICAL_FEATURE_RANGES["Eta"])
            ax.set_ylim(*PHYSICAL_FEATURE_RANGES["Phi"])
            ax.grid(alpha=0.2)
            _set_title(ax, f"{group_name} event {column + 1}")
            if row == len(event_groups) - 1:
                ax.set_xlabel(r"$\eta$")
            if column == 0:
                ax.set_ylabel(r"$\phi$")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=8,
            linestyle="none",
            markerfacecolor="grey",
            markeredgecolor="black",
            markeredgewidth=0.8,
            alpha=0.75,
            label="Original",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            markersize=9,
            linestyle="none",
            color="black",
            markeredgewidth=1.8,
            label="Reconstructed",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncols=2,
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_label(r"$\log_{10}(p_{\mathrm{T}} / \mathrm{GeV})$")
    return fig


def _hist_bins(original: np.ndarray, reconstructed: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.concatenate([original, reconstructed])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if min_value == max_value:
        width = max(abs(min_value) * 0.1, 1e-3)
        min_value -= width
        max_value += width
    return np.linspace(min_value, max_value, n_bins + 1)


def collect_reconstruction_histograms(
    feature_names: list[str],
    x_np: np.ndarray,
    x_hat_np: np.ndarray,
    mask_np: np.ndarray,
    n_bins: int = 50,
) -> dict[str, np.ndarray]:
    """Collect per-feature reconstruction histograms from padded model arrays."""
    histograms = {}
    for i, feature_name in enumerate(feature_names):
        original, reconstructed = _finite_pair(
            _masked_values(x_np[..., i], mask_np),
            _masked_values(x_hat_np[..., i], mask_np),
        )
        bins = _linear_bins(
            TRANSFORMED_FEATURE_RANGES.get(feature_name, (-1.0, 1.0)),
            n_bins,
        )
        clean_name = _clean_feature_name(feature_name)
        histograms[f"{clean_name}_orig_counts"] = _density_hist_with_overflow(original, bins)
        histograms[f"{clean_name}_reco_counts"] = _density_hist_with_overflow(
            reconstructed,
            bins,
        )
        histograms[f"{clean_name}_bins"] = bins
        diff_bins = _linear_bins(TRANSFORMED_RESIDUAL_RANGE, n_bins)
        histograms[f"{clean_name}_diff_counts"] = _density_hist_with_overflow(
            reconstructed - original,
            diff_bins,
        )
        histograms[f"{clean_name}_diff_bins"] = diff_bins
    return histograms


def _hist_step(ax, bins, counts, label=None, color=RECONSTRUCTED_COLOR):
    ax.stairs(
        counts,
        bins,
        label=label,
        color=color,
        linewidth=HISTOGRAM_LINEWIDTH,
    )


def _hist_fill(ax, bins, counts, label=None, color=ORIGINAL_COLOR):
    ax.stairs(
        counts,
        bins,
        label=label,
        color=color,
        fill=True,
        alpha=HISTOGRAM_FILL_ALPHA,
        linewidth=0,
    )


def _plot_original_reconstructed_histograms(ax, bins, original, reconstructed):
    _hist_fill(ax, bins, original, label="Original", color=ORIGINAL_COLOR)
    _hist_step(
        ax,
        bins,
        reconstructed,
        label="Reconstructed",
        color=RECONSTRUCTED_COLOR,
    )


def _plot_ratio_histogram(
    ax,
    bins,
    numerator,
    denominator,
    label=None,
    color=RECONSTRUCTED_COLOR,
):
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    _hist_step(ax, bins, ratio, label=label, color=color)


def _single_run_residual_series(counts):
    return [(counts, None, RESIDUAL_COLOR)]


def _triplet_axes(fig, ratio_indices=(0,)):
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=RATIO_HEIGHT_RATIOS,
        hspace=RATIO_HSPACE,
        wspace=RATIO_WSPACE,
    )
    axes = []
    ratio_axes = {}
    for i in range(3):
        if i in ratio_indices:
            axis = fig.add_subplot(grid[0, i])
            ratio_axes[i] = fig.add_subplot(grid[1, i], sharex=axis)
        else:
            axis = fig.add_subplot(grid[:, i])
        axes.append(axis)
    return axes, ratio_axes


def _pair_axes(fig, with_first_ratio=False):
    if not with_first_ratio:
        return fig.subplots(1, 2), {}

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=RATIO_HEIGHT_RATIOS,
        hspace=RATIO_HSPACE,
        wspace=RATIO_WSPACE,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[:, 1]),
    ]
    return axes, {0: fig.add_subplot(grid[1, 0], sharex=axes[0])}


def _configure_ratio_axis(ax, xlabel):
    ax.axhline(
        1.0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.set_ylim(*RATIO_YLIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reco / orig")


def _adjust_ratio_layout(fig):
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.88)


def _grid(n_items: int, max_cols: int = 3, figsize_per_axis=(5.0, 3.6)):
    n_cols = min(max_cols, max(n_items, 1))
    n_rows = math.ceil(max(n_items, 1) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_axis[0] * n_cols, figsize_per_axis[1] * n_rows),
        squeeze=False,
    )
    return fig, axes.flatten()


def plot_feature_histograms(
    histograms: dict[str, np.ndarray],
    feature_names: list[str],
    mse_per_feature: np.ndarray | None = None,
    title: str = "Original vs. reconstructed features",
):
    plot_utils.set_mpl_style()
    fig, axes = _grid(len(feature_names))
    _set_suptitle(fig, title, fontsize=16)
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        bins = histograms[f"{clean_name}_bins"]
        _hist_fill(
            ax,
            bins,
            histograms[f"{clean_name}_orig_counts"],
            label="Original",
            color=ORIGINAL_COLOR,
        )
        _hist_step(
            ax,
            bins,
            histograms[f"{clean_name}_reco_counts"],
            label="Reconstructed",
            color=RECONSTRUCTED_COLOR,
        )
        metric = ""
        if mse_per_feature is not None:
            metric = f" (MSE: {mse_per_feature[i]:.4g})"
        _set_title(ax, f"{_feature_label(feature_name)}{metric}")
        ax.set_xlabel(_feature_label(feature_name))
        ax.set_ylabel("Density")
        ax.legend()
        if feature_name.lower() in {"pt", "p_t"} or feature_name.endswith("_PT"):
            ax.set_yscale("log", nonpositive="clip")
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_residual_histograms(
    histograms: dict[str, np.ndarray],
    feature_names: list[str],
    title: str = "Reconstructed minus original",
):
    plot_utils.set_mpl_style()
    fig, axes = _grid(len(feature_names))
    _set_suptitle(fig, title, fontsize=16)
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        _hist_step(
            ax,
            histograms[f"{clean_name}_diff_bins"],
            histograms[f"{clean_name}_diff_counts"],
            color=RESIDUAL_COLOR,
        )
        ax.axvline(
            0.0,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
        )
        _set_title(ax, _feature_label(feature_name))
        ax.set_xlabel(f"Reco - original\n{_feature_label(feature_name)}")
        ax.set_ylabel("Density")
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _validate_data_level(data_level: str) -> None:
    if data_level not in ("particle", "jet"):
        raise ValueError(f"Unsupported data level: {data_level}")


def _safe_density_hist(values, bins):
    return _density_hist_with_overflow(values, bins)


def _physical_feature_bins(
    feature_name: str,
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
):
    ranges = feature_ranges or PHYSICAL_FEATURE_RANGES
    return _linear_bins(ranges.get(feature_name, (0.0, 1.0)), 50)


def collect_physical_reconstruction_histograms(
    feature_names: list[str],
    x_np: np.ndarray,
    x_hat_np: np.ndarray,
    true_jet_pts=(),
    reco_jet_pts=(),
    true_jet_masses=(),
    reco_jet_masses=(),
    true_tau32s=(),
    reco_tau32s=(),
    true_missing_ets=(),
    reco_missing_ets=(),
    true_jet_etas=(),
    reco_jet_etas=(),
    true_jet_phis=(),
    reco_jet_phis=(),
    unfiltered_true_jet_pts=(),
    unfiltered_reco_jet_pts=(),
    unfiltered_true_jet_masses=(),
    unfiltered_reco_jet_masses=(),
    unfiltered_true_tau32s=(),
    unfiltered_reco_tau32s=(),
    unfiltered_true_jet_etas=(),
    unfiltered_reco_jet_etas=(),
    unfiltered_true_jet_phis=(),
    unfiltered_reco_jet_phis=(),
    data_level: str = "particle",
    physical_feature_ranges: dict[str, tuple[float, float]] | None = None,
    missing_et_range: tuple[float, float] = MISSING_ET_RANGE,
    jet_mass_range: tuple[float, float] = JET_MASS_RANGE,
) -> dict[str, np.ndarray]:
    """Collect ORBIT paper-style histograms in physical coordinates.

    `x_np` and `x_hat_np` are flattened valid objects with columns
    `[Eta, Phi, pT]`.
    """
    _validate_data_level(data_level)
    histograms = {}

    if len(true_jet_pts) > 0:
        true_jet_pts = np.asarray(true_jet_pts)
        reco_jet_pts = np.asarray(reco_jet_pts)
        fractional_diff = (reco_jet_pts - true_jet_pts) / (true_jet_pts + 1e-8)
        bins = _linear_bins((-0.5, 0.5), 50)
        histograms["jet_pt_resolution_counts"] = _counts_hist_with_overflow(
            fractional_diff,
            bins,
        )
        histograms["jet_pt_resolution_bins"] = bins

    if len(unfiltered_true_jet_pts) > 0:
        unfiltered_true_jet_pts = np.asarray(unfiltered_true_jet_pts)
        unfiltered_reco_jet_pts = np.asarray(unfiltered_reco_jet_pts)
        fractional_diff = (
            unfiltered_reco_jet_pts - unfiltered_true_jet_pts
        ) / (unfiltered_true_jet_pts + 1e-8)
        bins = _linear_bins((-0.5, 0.5), 50)
        histograms["jet_pt_resolution_unfiltered_counts"] = (
            _counts_hist_with_overflow(fractional_diff, bins)
        )
        histograms["jet_pt_resolution_unfiltered_bins"] = bins

    if len(true_jet_etas) > 0:
        true_jet_etas = np.asarray(true_jet_etas)
        reco_jet_etas = np.asarray(reco_jet_etas)
        diff = reco_jet_etas - true_jet_etas
        bins = _linear_bins(PHYSICAL_RESIDUAL_RANGES["Eta"], 50)
        histograms["jet_eta_resolution_counts"] = _counts_hist_with_overflow(diff, bins)
        histograms["jet_eta_resolution_bins"] = bins

    if len(unfiltered_true_jet_etas) > 0:
        diff = np.asarray(unfiltered_reco_jet_etas) - np.asarray(
            unfiltered_true_jet_etas
        )
        bins = _linear_bins(PHYSICAL_RESIDUAL_RANGES["Eta"], 50)
        histograms["jet_eta_resolution_unfiltered_counts"] = (
            _counts_hist_with_overflow(diff, bins)
        )
        histograms["jet_eta_resolution_unfiltered_bins"] = bins

    if len(true_jet_phis) > 0:
        true_jet_phis = np.asarray(true_jet_phis)
        reco_jet_phis = np.asarray(reco_jet_phis)
        diff = _angular_difference(reco_jet_phis, true_jet_phis)
        bins = _linear_bins(PHYSICAL_RESIDUAL_RANGES["Phi"], 50)
        histograms["jet_phi_resolution_counts"] = _counts_hist_with_overflow(diff, bins)
        histograms["jet_phi_resolution_bins"] = bins

    if len(unfiltered_true_jet_phis) > 0:
        diff = _angular_difference(
            np.asarray(unfiltered_reco_jet_phis),
            np.asarray(unfiltered_true_jet_phis),
        )
        bins = _linear_bins(PHYSICAL_RESIDUAL_RANGES["Phi"], 50)
        histograms["jet_phi_resolution_unfiltered_counts"] = (
            _counts_hist_with_overflow(diff, bins)
        )
        histograms["jet_phi_resolution_unfiltered_bins"] = bins

    for i, feature_name in enumerate(feature_names):
        original, reconstructed = _finite_pair(x_np[:, i], x_hat_np[:, i])
        bins = _physical_feature_bins(
            feature_name,
            original,
            reconstructed,
            feature_ranges=physical_feature_ranges,
        )
        clean_name = _clean_feature_name(feature_name)
        histograms[f"{clean_name}_orig_counts"] = _safe_density_hist(original, bins)
        histograms[f"{clean_name}_reco_counts"] = _safe_density_hist(reconstructed, bins)
        histograms[f"{clean_name}_bins"] = bins
        diff = (
            _angular_difference(reconstructed, original)
            if feature_name == "Phi"
            else reconstructed - original
        )
        diff_bins = _linear_bins(PHYSICAL_RESIDUAL_RANGES.get(feature_name, (-1.0, 1.0)), 50)
        histograms[f"{clean_name}_diff_counts"] = _density_hist_with_overflow(diff, diff_bins)
        histograms[f"{clean_name}_diff_bins"] = diff_bins

    energy_orig = x_np[:, 2] * np.cosh(x_np[:, 0])
    energy_reco = x_hat_np[:, 2] * np.cosh(x_hat_np[:, 0])
    finite_energy = np.isfinite(energy_orig) & np.isfinite(energy_reco)
    energy_orig = energy_orig[finite_energy]
    energy_reco = energy_reco[finite_energy]
    if energy_orig.size > 0:
        energy_bins = _linear_bins(ENERGY_RANGE, 50)
        histograms["energy_orig_counts"] = _safe_density_hist(energy_orig, energy_bins)
        histograms["energy_reco_counts"] = _safe_density_hist(energy_reco, energy_bins)
        histograms["energy_bins"] = energy_bins
        energy_residual_bins = _linear_bins(ENERGY_RESIDUAL_RANGE, 50)
        histograms["energy_residuals_counts"] = _density_hist_with_overflow(
            energy_reco - energy_orig,
            energy_residual_bins,
        )
        histograms["energy_residuals_bins"] = energy_residual_bins

    if data_level == "particle" and len(true_missing_ets) > 0:
        true_missing_ets = np.asarray(true_missing_ets)
        reco_missing_ets = np.asarray(reco_missing_ets)
        missing_et_bins = _linear_bins(missing_et_range, 50)
        histograms["missing_et_orig_counts"] = _safe_density_hist(
            true_missing_ets,
            missing_et_bins,
        )
        histograms["missing_et_reco_counts"] = _safe_density_hist(
            reco_missing_ets,
            missing_et_bins,
        )
        histograms["missing_et_bins"] = missing_et_bins

    if data_level == "particle" and len(true_jet_masses) > 0:
        true_jet_masses = np.asarray(true_jet_masses)
        reco_jet_masses = np.asarray(reco_jet_masses)

        mass_bins = _linear_bins(jet_mass_range, 50)
        histograms["jet_mass_orig_counts"] = _safe_density_hist(
            true_jet_masses,
            mass_bins,
        )
        histograms["jet_mass_reco_counts"] = _safe_density_hist(
            reco_jet_masses,
            mass_bins,
        )
        histograms["jet_mass_bins"] = mass_bins

        mass_diff_bins = _linear_bins(JET_MASS_RESIDUAL_RANGE, 50)
        histograms["jet_mass_diff_counts"] = _safe_density_hist(
            reco_jet_masses - true_jet_masses,
            mass_diff_bins,
        )
        histograms["jet_mass_diff_bins"] = mass_diff_bins

    if data_level == "particle" and len(unfiltered_true_jet_masses) > 0:
        unfiltered_true_jet_masses = np.asarray(unfiltered_true_jet_masses)
        unfiltered_reco_jet_masses = np.asarray(unfiltered_reco_jet_masses)
        mass_diff_bins = _linear_bins(JET_MASS_RESIDUAL_RANGE, 50)
        histograms["jet_mass_diff_unfiltered_counts"] = _safe_density_hist(
            unfiltered_reco_jet_masses - unfiltered_true_jet_masses,
            mass_diff_bins,
        )
        histograms["jet_mass_diff_unfiltered_bins"] = mass_diff_bins

    if data_level == "particle" and len(true_tau32s) > 0:
        true_tau32s = np.asarray(true_tau32s)
        reco_tau32s = np.asarray(reco_tau32s)
        tau_diff_bins = _linear_bins(TAU32_RESIDUAL_RANGE, 50)
        histograms["tau32_diff_counts"] = _safe_density_hist(
            reco_tau32s - true_tau32s,
            tau_diff_bins,
        )
        histograms["tau32_diff_bins"] = tau_diff_bins

    if data_level == "particle" and len(unfiltered_true_tau32s) > 0:
        tau_diff_bins = _linear_bins(TAU32_RESIDUAL_RANGE, 50)
        histograms["tau32_diff_unfiltered_counts"] = _safe_density_hist(
            np.asarray(unfiltered_reco_tau32s) - np.asarray(unfiltered_true_tau32s),
            tau_diff_bins,
        )
        histograms["tau32_diff_unfiltered_bins"] = tau_diff_bins

    return histograms


def plot_physical_residual_histogram(
    ax,
    bins,
    series,
    xlabel,
    title=None,
    ylabel="Density",
):
    for counts, label, color in series:
        _hist_step(ax, bins, counts, label=label, color=color)
    ax.axvline(
        0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        alpha=REFERENCE_LINE_ALPHA,
    )
    if title:
        _set_title(ax, title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if any(label for _, label, _ in series):
        ax.legend(prop={"size": 10})


def plot_physical_feature_histograms(
    histograms: dict[str, np.ndarray],
    feature_names: list[str],
    reconstructed_series=None,
    mse_per_feature=None,
    include_all_ratios: bool = False,
    title: str = "Original vs. Reconstructed Features",
):
    plot_utils.set_mpl_style()
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig = plt.figure(figsize=FEATURE_FIGSIZE)
    ratio_indices = (0, 1, 2) if include_all_ratios else (0,)
    axes, ratio_axes = _triplet_axes(fig, ratio_indices=ratio_indices)
    _set_suptitle(fig, title, fontsize=16)

    for axis_idx, feature_idx in enumerate((2, 0, 1)):
        axis = axes[axis_idx]
        feature_name = feature_names[feature_idx]
        clean_name = _clean_feature_name(feature_name)
        bins = histograms[f"{clean_name}_bins"]
        if single_run:
            _plot_original_reconstructed_histograms(
                axis,
                bins,
                histograms[f"{clean_name}_orig_counts"],
                histograms[f"{clean_name}_reco_counts"],
            )
        else:
            _hist_fill(
                axis,
                bins,
                histograms[f"{clean_name}_orig_counts"],
                label="Original (Truth)",
                color=TRUTH_REFERENCE_COLOR,
            )
        for data, label, color in reconstructed_series:
            if not single_run:
                _hist_step(
                    axis,
                    bins,
                    data[f"{clean_name}_reco_counts"],
                    label=label,
                    color=color,
                )
            if axis_idx in ratio_axes:
                _plot_ratio_histogram(
                    ratio_axes[axis_idx],
                    bins,
                    data[f"{clean_name}_reco_counts"],
                    histograms[f"{clean_name}_orig_counts"],
                    label=label,
                    color=color,
                )

        metric = ""
        if mse_per_feature is not None:
            metric = f" (MSE: {mse_per_feature[feature_idx]:.4f})"
        _set_title(axis, f"{feature_name}{metric}")
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Density")
        axis.legend(prop={"size": 10})
        if feature_name == "pT":
            axis.set_yscale("log", nonpositive="clip")
        if axis_idx in ratio_axes:
            _configure_ratio_axis(ratio_axes[axis_idx], xlabel=feature_name)
            axis.tick_params(labelbottom=False)

    _adjust_ratio_layout(fig)
    return fig


def plot_energy_histograms(
    histograms: dict[str, np.ndarray],
    reconstructed_series=None,
    include_ratio: bool = False,
    title: str = "Original vs. Reconstructed Energy (m=0)",
):
    plot_utils.set_mpl_style()
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig = plt.figure(figsize=ENERGY_FIGSIZE)
    axes, ratio_axes = _pair_axes(fig, with_first_ratio=include_ratio)
    bins = histograms["energy_bins"]
    if single_run:
        _plot_original_reconstructed_histograms(
            axes[0],
            bins,
            histograms["energy_orig_counts"],
            histograms["energy_reco_counts"],
        )
    else:
        _hist_fill(
            axes[0],
            bins,
            histograms["energy_orig_counts"],
            label="Original (Truth)",
            color=TRUTH_REFERENCE_COLOR,
        )
    for data, label, color in reconstructed_series:
        if not single_run:
            _hist_step(axes[0], bins, data["energy_reco_counts"], label=label, color=color)
        if include_ratio:
            _plot_ratio_histogram(
                ratio_axes[0],
                bins,
                data["energy_reco_counts"],
                histograms["energy_orig_counts"],
                label=label,
                color=color,
            )
    axes[0].set_yscale("log", nonpositive="clip")
    _set_title(axes[0], title)
    axes[0].set_xlabel("Energy [GeV]")
    axes[0].set_ylabel("Density")
    axes[0].legend(prop={"size": 10})
    if include_ratio:
        _configure_ratio_axis(ratio_axes[0], xlabel="Energy [GeV]")
        axes[0].tick_params(labelbottom=False)

    plot_physical_residual_histogram(
        axes[1],
        histograms["energy_residuals_bins"],
        _single_run_residual_series(histograms["energy_residuals_counts"]),
        xlabel=r"$E^\mathrm{reco} - E^\mathrm{orig}$ [GeV]",
        title=r"Energy Residuals: $E^\mathrm{reco} - E^\mathrm{orig}$",
    )
    if include_ratio:
        _adjust_ratio_layout(fig)
    else:
        plt.tight_layout()
    return fig


def plot_missing_transverse_energy(histograms: dict[str, np.ndarray]):
    plot_utils.set_mpl_style()
    fig, ax = plt.subplots(figsize=RESOLUTION_FIGSIZE)
    bins = histograms["missing_et_bins"]
    _plot_original_reconstructed_histograms(
        ax,
        bins,
        histograms["missing_et_orig_counts"],
        histograms["missing_et_reco_counts"],
    )
    ax.set_xlabel(r"Missing transverse energy $E_T^\mathrm{miss}$ [GeV]")
    ax.set_ylabel("Density")
    ax.set_yscale("log", nonpositive="clip")
    _set_title(ax, "Missing Transverse Energy")
    ax.legend(prop={"size": 10})
    plt.tight_layout()
    return fig


def plot_paper_kinematic_distributions(
    histograms: dict[str, np.ndarray],
    data_level: str,
    reconstructed_series=None,
):
    plot_utils.set_mpl_style()
    _validate_data_level(data_level)
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig, axes = plt.subplots(1, 3, figsize=MONEY_TRIPLET_FIGSIZE)

    for axis, feature_name in zip(axes, ("pT", "Eta", "Phi")):
        feature_label = KINEMATIC_LABELS[feature_name]
        bins = histograms[f"{feature_name}_bins"]
        original = histograms[f"{feature_name}_orig_counts"]
        if single_run:
            _plot_original_reconstructed_histograms(
                axis,
                bins,
                original,
                histograms[f"{feature_name}_reco_counts"],
            )
        else:
            _hist_fill(
                axis,
                bins,
                original,
                label="Original (Truth)",
                color=TRUTH_REFERENCE_COLOR,
            )
            for data, label, color in reconstructed_series:
                _hist_step(
                    axis,
                    bins,
                    data[f"{feature_name}_reco_counts"],
                    label=label,
                    color=color,
                )
        axis.set_xlabel(feature_label)
        axis.set_ylabel("Density")
        if feature_name == "pT":
            axis.set_yscale("log", nonpositive="clip")
        _set_title(axis, f"{feature_label} distribution")
        axis.legend(prop={"size": 10})

    _set_suptitle(fig, f"{data_level.capitalize()} kinematics: original vs. reconstructed")
    plt.tight_layout()
    return fig


def plot_paper_kinematic_differences(
    histograms: dict[str, np.ndarray],
    data_level: str,
    difference_series=None,
):
    plot_utils.set_mpl_style()
    _validate_data_level(data_level)
    difference_series = difference_series or [(histograms, None, RESIDUAL_COLOR)]
    fig, axes = plt.subplots(1, 3, figsize=MONEY_TRIPLET_FIGSIZE)

    for axis, feature_name in zip(axes, ("pT", "Eta", "Phi")):
        feature_label = KINEMATIC_LABELS[feature_name]
        plot_physical_residual_histogram(
            axis,
            histograms[f"{feature_name}_diff_bins"],
            [
                (data[f"{feature_name}_diff_counts"], label, color)
                for data, label, color in difference_series
            ],
            xlabel=KINEMATIC_RESIDUAL_LABELS[feature_name],
            title=f"{feature_label} residuals",
        )
        axis.set_yscale("log", nonpositive="clip")

    _set_suptitle(fig, f"{data_level.capitalize()} kinematic residuals")
    plt.tight_layout()
    return fig


def paper_reconstruction_plots(
    histograms: dict[str, np.ndarray],
    data_level: str,
) -> dict[str, object]:
    return {
        "paper_kinematic_distributions": plot_paper_kinematic_distributions(
            histograms,
            data_level,
        ),
        "paper_kinematic_differences": plot_paper_kinematic_differences(
            histograms,
            data_level,
        ),
    }


def physical_reconstruction_plots(
    feature_names: list[str],
    mse_per_feature: np.ndarray,
    histograms: dict[str, np.ndarray],
    data_level: str = "particle",
    include_all_ratios: bool = False,
    jet_matching_cut_label: str = r"With $\Delta R$ cutoff",
) -> dict[str, object]:
    """Build ORBIT paper and exploratory single-run reconstruction figures."""
    figures = paper_reconstruction_plots(histograms, data_level)

    has_pt_res = "jet_pt_resolution_counts" in histograms
    has_eta_res = "jet_eta_resolution_counts" in histograms
    has_phi_res = "jet_phi_resolution_counts" in histograms
    if has_pt_res and (has_eta_res or has_phi_res):
        fig, axes = plt.subplots(1, 3, figsize=SUBSTRUCTURE_SIMPLE_FIGSIZE)
        _set_suptitle(fig, "Jet Kinematic Resolution", fontsize=16)
        _hist_step(
            axes[0],
            histograms["jet_pt_resolution_bins"],
            histograms["jet_pt_resolution_counts"],
            label=jet_matching_cut_label,
            color=RESIDUAL_COLOR,
        )
        if "jet_pt_resolution_unfiltered_counts" in histograms:
            _hist_step(
                axes[0],
                histograms["jet_pt_resolution_unfiltered_bins"],
                histograms["jet_pt_resolution_unfiltered_counts"],
                label=r"No $\Delta R$ cutoff",
                color=RUN_COLORS[3],
            )
        axes[0].axvline(0, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                        alpha=REFERENCE_LINE_ALPHA)
        axes[0].set_xlabel(r"$(p_T^\mathrm{reco} - p_T^\mathrm{true}) / p_T^\mathrm{true}$")
        axes[0].set_ylabel("Number of Jets")
        _set_title(axes[0], r"Fractional $p_T$ Resolution")
        axes[0].legend()

        if has_eta_res:
            _hist_step(
                axes[1],
                histograms["jet_eta_resolution_bins"],
                histograms["jet_eta_resolution_counts"],
                label=jet_matching_cut_label,
                color=RESIDUAL_COLOR,
            )
            if "jet_eta_resolution_unfiltered_counts" in histograms:
                _hist_step(
                    axes[1],
                    histograms["jet_eta_resolution_unfiltered_bins"],
                    histograms["jet_eta_resolution_unfiltered_counts"],
                    label=r"No $\Delta R$ cutoff",
                    color=RUN_COLORS[3],
                )
            axes[1].axvline(0, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                            alpha=REFERENCE_LINE_ALPHA)
            axes[1].set_xlabel(r"$\eta^\mathrm{reco} - \eta^\mathrm{true}$")
            axes[1].set_ylabel("Number of Jets")
            _set_title(axes[1], r"$\eta$ Residual")
            axes[1].legend()
        else:
            axes[1].axis("off")

        if has_phi_res:
            _hist_step(
                axes[2],
                histograms["jet_phi_resolution_bins"],
                histograms["jet_phi_resolution_counts"],
                label=jet_matching_cut_label,
                color=RESIDUAL_COLOR,
            )
            if "jet_phi_resolution_unfiltered_counts" in histograms:
                _hist_step(
                    axes[2],
                    histograms["jet_phi_resolution_unfiltered_bins"],
                    histograms["jet_phi_resolution_unfiltered_counts"],
                    label=r"No $\Delta R$ cutoff",
                    color=RUN_COLORS[3],
                )
            axes[2].axvline(0, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                            alpha=REFERENCE_LINE_ALPHA)
            axes[2].set_xlabel(r"$\phi^\mathrm{reco} - \phi^\mathrm{true}$")
            axes[2].set_ylabel("Number of Jets")
            _set_title(axes[2], r"$\phi$ Residual")
            axes[2].legend()
        else:
            axes[2].axis("off")

        plt.tight_layout()
        figures["jet_kinematic_resolution"] = fig
    elif has_pt_res:
        fig, ax = plt.subplots(figsize=RESOLUTION_FIGSIZE)
        _hist_step(
            ax,
            histograms["jet_pt_resolution_bins"],
            histograms["jet_pt_resolution_counts"],
            label=jet_matching_cut_label,
            color=RESIDUAL_COLOR,
        )
        if "jet_pt_resolution_unfiltered_counts" in histograms:
            _hist_step(
                ax,
                histograms["jet_pt_resolution_unfiltered_bins"],
                histograms["jet_pt_resolution_unfiltered_counts"],
                label=r"No $\Delta R$ cutoff",
                color=RUN_COLORS[3],
            )
        ax.axvline(0, color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
                   alpha=REFERENCE_LINE_ALPHA)
        ax.set_xlabel(
            r"Fractional $p_T$ Resolution: "
            r"$(p_T^\mathrm{reco} - p_T^\mathrm{true}) / p_T^\mathrm{true}$"
        )
        ax.set_ylabel("Number of Jets")
        _set_title(ax, "Jet Transverse Momentum Recovery")
        ax.legend()
        figures["jet_pt_resolution"] = fig

    figures["kinematics"] = plot_physical_feature_histograms(
        histograms,
        feature_names,
        mse_per_feature=mse_per_feature,
        include_all_ratios=include_all_ratios,
        title=f"{data_level.capitalize()} Kinematics: Original vs. Reconstructed",
    )
    if "energy_bins" in histograms:
        figures["energy_residuals"] = plot_energy_histograms(
            histograms,
            include_ratio=include_all_ratios,
            title=f"{data_level.capitalize()} Energy: Original vs. Reconstructed (m=0)",
        )
    if data_level == "particle" and "missing_et_bins" in histograms:
        figures["paper_missing_transverse_energy"] = plot_missing_transverse_energy(
            histograms,
        )
    if data_level == "particle" and "jet_mass_orig_counts" in histograms:
        include_tau32 = "tau32_diff_counts" in histograms
        n_columns = 3 if include_tau32 else 2
        figsize = SUBSTRUCTURE_SIMPLE_FIGSIZE if include_tau32 else (12, 4)
        fig, axes = plt.subplots(1, n_columns, figsize=figsize)
        _set_suptitle(fig, "Jet Substructure", fontsize=16)
        _plot_original_reconstructed_histograms(
            axes[0],
            histograms["jet_mass_bins"],
            histograms["jet_mass_orig_counts"],
            histograms["jet_mass_reco_counts"],
        )
        _set_title(axes[0], "Jet Mass")
        axes[0].set_xlabel("Jet Mass [GeV]")
        axes[0].set_ylabel("Density")
        axes[0].set_yscale("log", nonpositive="clip")
        axes[0].legend()

        mass_residual_series = [
            (
                histograms["jet_mass_diff_counts"],
                jet_matching_cut_label,
                RESIDUAL_COLOR,
            )
        ]
        if "jet_mass_diff_unfiltered_counts" in histograms:
            mass_residual_series.append(
                (
                    histograms["jet_mass_diff_unfiltered_counts"],
                    r"No $\Delta R$ cutoff",
                    RUN_COLORS[3],
                )
            )
        plot_physical_residual_histogram(
            axes[1],
            histograms["jet_mass_diff_bins"],
            mass_residual_series,
            xlabel=r"$m^\mathrm{reco} - m^\mathrm{orig}$ [GeV]",
            title="Jet Mass Residuals",
        )
        if include_tau32:
            tau_residual_series = [
                (
                    histograms["tau32_diff_counts"],
                    jet_matching_cut_label,
                    RESIDUAL_COLOR,
                )
            ]
            if "tau32_diff_unfiltered_counts" in histograms:
                tau_residual_series.append(
                    (
                        histograms["tau32_diff_unfiltered_counts"],
                        r"No $\Delta R$ cutoff",
                        RUN_COLORS[3],
                    )
                )
            plot_physical_residual_histogram(
                axes[2],
                histograms["tau32_diff_bins"],
                tau_residual_series,
                xlabel=r"$\tau_{32}^\mathrm{reco} - \tau_{32}^\mathrm{orig}$",
                title=r"$\tau_{32}$ Residuals",
            )
        plt.tight_layout()
        figures["jet_substructure"] = fig

    return figures


def plot_codebook_histogram(code_idx: np.ndarray, num_codes: int | None = None):
    plot_utils.set_mpl_style()
    codes = np.asarray(code_idx).reshape(-1)
    codes = codes[np.isfinite(codes)]
    fig, ax = plt.subplots(figsize=(8, 5))
    xlabel = "Token ID"
    if codes.size == 0:
        ax.hist(codes, bins=np.arange(2) - 0.5, color=RECONSTRUCTED_COLOR, histtype="step")
    else:
        unique_codes, counts = np.unique(codes.astype(np.int64), return_counts=True)
        dense_safe = (
            num_codes is not None
            and num_codes <= 4096
            and unique_codes.max(initial=0) <= 4096
        )
        if dense_safe:
            bins = np.arange(int(unique_codes.max()) + 2) - 0.5
            ax.hist(codes, bins=bins, color=RECONSTRUCTED_COLOR, histtype="step")
        else:
            order = np.argsort(counts)[::-1]
            max_bars = 100
            selected = order[:max_bars]
            ax.bar(
                np.arange(len(selected)),
                counts[selected],
                color=RECONSTRUCTED_COLOR,
                alpha=HISTOGRAM_FILL_ALPHA,
            )
            ax.set_xticks([])
            xlabel = f"Active token ID, top {len(selected)} by count"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    utilization_denominator = num_codes or max(len(np.unique(codes)), 1)
    utilization = len(np.unique(codes)) / utilization_denominator
    _set_title(ax, f"Codebook utilization: {utilization:.3f}")
    fig.tight_layout()
    return fig


def attention_delta_eta_phi_figure(
    weights: torch.Tensor,
    x: torch.Tensor,
    valid: torch.Tensor,
    title: str,
    exclude_self: bool = False,
):
    """Plot attention versus angular distance.

    For absolute ORBIT features this expects feature order
    `[eta_scaled, cos(phi), sin(phi), pt_transformed]`.
    """
    attn = weights.mean(dim=1)
    eta = x[..., 0] * 3.0
    phi = torch.atan2(x[..., 2], x[..., 1])

    deta = eta[:, :, None] - eta[:, None, :]
    dphi = phi[:, :, None] - phi[:, None, :]
    dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi

    pair_mask = valid[:, :, None] & valid[:, None, :]
    if exclude_self:
        self_mask = torch.eye(pair_mask.shape[-1], dtype=torch.bool, device=pair_mask.device)
        pair_mask = pair_mask & ~self_mask[None, :, :]
    if not pair_mask.any():
        return None

    deta_np = deta[pair_mask].detach().cpu().numpy()
    dphi_np = dphi[pair_mask].detach().cpu().numpy()
    weight_np = attn[pair_mask].detach().cpu().numpy()

    weight_sum, eta_edges, phi_edges = np.histogram2d(
        deta_np,
        dphi_np,
        bins=(60, 64),
        range=((-6.0, 6.0), (-math.pi, math.pi)),
        weights=weight_np,
    )
    pair_count, _, _ = np.histogram2d(deta_np, dphi_np, bins=(eta_edges, phi_edges))
    hist = np.divide(
        weight_sum,
        pair_count,
        out=np.zeros_like(weight_sum),
        where=pair_count > 0,
    )

    plot_utils.set_mpl_style()
    fig, ax = plt.subplots(figsize=ATTENTION_DELTA_FIGSIZE)
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[eta_edges[0], eta_edges[-1], phi_edges[0], phi_edges[-1]],
        aspect="auto",
    )
    _set_title(ax, title)
    ax.set_xlabel(r"$\Delta\eta = \eta_\mathrm{query} - \eta_\mathrm{key}$")
    ax.set_ylabel(r"$\Delta\phi = \phi_\mathrm{query} - \phi_\mathrm{key}$")
    fig.colorbar(im, ax=ax, label="mean attention weight per pair")
    return fig


def attention_map_figure(matrix, title: str):
    plot_utils.set_mpl_style()
    fig, ax = plt.subplots(figsize=ATTENTION_MAP_FIGSIZE)
    im = ax.imshow(matrix, vmin=0.0, vmax=max(float(np.max(matrix)), 1e-6), aspect="auto")
    _set_title(ax, title)
    ax.set_xlabel("key token")
    ax.set_ylabel("query token")
    fig.colorbar(im, ax=ax)
    return fig


def close_figure(fig) -> None:
    plt.close(fig)


def _multirun_mplhep():
    """Apply the shared HEP plotting style used by all multirun figures."""
    return plot_utils.set_mpl_style()


def codebook_marker_areas(
    records: list[dict], size_range: tuple[float, float] | None = None
) -> np.ndarray:
    """Map effective codebook size to a bounded, logarithmic marker area."""
    sizes = np.asarray(
        [record.get("total_codebook_size") or np.nan for record in records],
        dtype=float,
    )
    finite = np.isfinite(sizes) & (sizes > 0)
    if not np.any(finite):
        return np.full(len(records), MIN_CODEBOOK_MARKER_AREA)
    log_sizes = np.log2(sizes[finite])
    if size_range is None:
        low, high = float(log_sizes.min()), float(log_sizes.max())
    else:
        low, high = map(float, np.log2(size_range))
    result = np.full(len(records), MIN_CODEBOOK_MARKER_AREA)
    if high > low:
        result[finite] += (MAX_CODEBOOK_MARKER_AREA - MIN_CODEBOOK_MARKER_AREA) * (
            (log_sizes - low) / (high - low)
        )
    else:
        result[finite] = 0.5 * (MIN_CODEBOOK_MARKER_AREA + MAX_CODEBOOK_MARKER_AREA)
    return result


def save_figures(figures: dict[str, object], output_dir: str | Path, suffix: str = "png") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(output_path / f"{name}.{suffix}", dpi=300, bbox_inches="tight")


def _multirun_feature_uses_log_y(feature_name: str) -> bool:
    lower_name = feature_name.lower()
    return "energy" in lower_name or "pt" in lower_name or "missing_et" in lower_name


def _multirun_feature_uses_log_x(feature_name: str) -> bool:
    return feature_name in {"physical_pT", "physical_energy"}


def plot_multirun_feature_histograms(
    runs: list[dict],
    feature_names: list[str],
    title: str = "Reconstruction comparison",
    figsize_per_axis: tuple[float, float] = (5.0, 3.6),
    show_titles: bool = True,
):
    """Overlay reconstructed feature histograms from several runs."""
    hep = _multirun_mplhep()
    fig, axes = _grid(len(feature_names), figsize_per_axis=figsize_per_axis)
    if show_titles:
        _set_suptitle(fig, title, fontsize=16)
    reference = runs[0]["histograms"]
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        hep.histplot(
            reference[f"{clean_name}_orig_counts"],
            reference[f"{clean_name}_bins"],
            ax=ax,
            label="Original",
            color=TRUTH_REFERENCE_COLOR,
            histtype="fill",
            alpha=TRUTH_REFERENCE_FILL_ALPHA,
        )
        for run in runs:
            histograms = run["histograms"]
            hep.histplot(
                histograms[f"{clean_name}_reco_counts"],
                histograms[f"{clean_name}_bins"],
                ax=ax,
                label=multirun_display_label(run["label"]),
                color=run.get("plot_color", multirun_color(run["label"])),
                histtype="step",
                linewidth=HISTOGRAM_LINEWIDTH,
            )
        if show_titles:
            _set_title(ax, _feature_label(feature_name))
        ax.set_xlabel(_feature_label(feature_name))
        ax.set_ylabel("Density")
        if _multirun_feature_uses_log_x(feature_name):
            ax.set_xscale("log")
            positive_bins = reference[f"{clean_name}_bins"][
                reference[f"{clean_name}_bins"] > 0
            ]
            if positive_bins.size > 0:
                ax.set_xlim(left=positive_bins[0])
        ax.set_yscale("log", nonpositive="clip")
        if show_titles:
            ax.legend(fontsize=MULTIRUN_LEGEND_FONTSIZE)
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    if not show_titles:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(5, len(labels)),
            fontsize=PRESENTATION_LEGEND_FONTSIZE,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    else:
        fig.tight_layout()
    return fig


def plot_multirun_residual_histograms(
    runs: list[dict],
    feature_names: list[str],
    title: str = "Reconstruction residual comparison",
    show_titles: bool = True,
    figsize_per_axis: tuple[float, float] = (5.0, 5.6),
):
    """Overlay reconstructed-minus-original residuals from several runs."""
    hep = _multirun_mplhep()
    fig, axes = _grid(len(feature_names), figsize_per_axis=figsize_per_axis)
    if show_titles:
        _set_suptitle(fig, title, fontsize=16)
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        clean_name = _clean_feature_name(feature_name)
        for run in runs:
            histograms = run["histograms"]
            hep.histplot(
                histograms[f"{clean_name}_diff_counts"],
                histograms[f"{clean_name}_diff_bins"],
                ax=ax,
                label=multirun_display_label(run["label"]),
                color=run.get("plot_color", multirun_color(run["label"])),
                histtype="step",
                linewidth=HISTOGRAM_LINEWIDTH,
            )
        ax.axvline(
            0.0,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
        )
        ax.set_yscale("log", nonpositive="clip")
        if show_titles:
            _set_title(ax, _feature_label(feature_name))
        ax.set_xlabel(f"Reco - original\n{_feature_label(feature_name)}")
        ax.set_ylabel("Density")
        if show_titles:
            ax.legend(fontsize=MULTIRUN_LEGEND_FONTSIZE)
    for ax in axes[len(feature_names) :]:
        ax.axis("off")
    if not show_titles:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(4, len(labels)),
            fontsize=PRESENTATION_LEGEND_FONTSIZE,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    else:
        fig.tight_layout()
    return fig


def plot_multirun_metric(
    records: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    log_x: bool = True,
    log_y: bool = False,
    x_metric: str = "total_codebook_size",
    xlabel: str = "Total codebook size",
    reference_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    reference_label: str | None = None,
    vertical_reference: float | None = None,
    vertical_reference_label: str | None = None,
    horizontal_reference: float | None = None,
    horizontal_reference_label: str | None = None,
    show_title: bool = True,
    scale_marker_by_codebook_size: bool = False,
    marker_area_scale: float = 1.0,
):
    """Plot one scalar metric against another, grouped by run family."""
    _multirun_mplhep()
    usable_records = [
        record
        for record in records
        if record.get(metric) is not None and record.get(x_metric) is not None
    ]
    if not usable_records:
        return None

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    grouped_records = defaultdict(list)
    for record in usable_records:
        grouped_records[record.get("plot_family") or record["label"]].append(record)

    explicit_marker_areas = all(
        record.get("plot_marker_area") is not None for record in usable_records
    )
    variable_marker_areas = scale_marker_by_codebook_size or explicit_marker_areas
    codebook_size_range = None
    if scale_marker_by_codebook_size and not explicit_marker_areas:
        codebook_sizes = np.asarray(
            [record.get("total_codebook_size") or np.nan for record in usable_records],
            dtype=float,
        )
        finite_sizes = codebook_sizes[
            np.isfinite(codebook_sizes) & (codebook_sizes > 0)
        ]
        if finite_sizes.size:
            codebook_size_range = (
                float(finite_sizes.min()),
                float(finite_sizes.max()),
            )

    for family, family_records in sorted(
        grouped_records.items(), key=lambda item: multirun_legend_sort_key(item[0])
    ):
        color = multirun_color(family)
        marker = multirun_marker(family)
        family_records = sorted(
            family_records,
            key=lambda record: (record[x_metric], str(record["label"])),
        )
        x_values = [record[x_metric] for record in family_records]
        y_values = [record[metric] for record in family_records]
        if len(family_records) > 1:
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.5,
                alpha=0.75,
                marker=None if variable_marker_areas else marker,
                markersize=7,
                label=None if variable_marker_areas else multirun_display_label(family),
            )
            scatter_label = multirun_display_label(family) if variable_marker_areas else None
        else:
            scatter_label = multirun_display_label(family)
        ax.scatter(
            x_values,
            y_values,
            s=(
                [record["plot_marker_area"] for record in family_records]
                if explicit_marker_areas
                else marker_area_scale
                * codebook_marker_areas(family_records, codebook_size_range)
                if scale_marker_by_codebook_size
                else None
            ),
            label=scatter_label,
            color=color,
            marker=marker,
            zorder=3,
        )
    if reference_fn is not None:
        reference_x = np.asarray(
            sorted({float(record[x_metric]) for record in usable_records}),
            dtype=np.float64,
        )
        ax.plot(
            reference_x,
            reference_fn(reference_x),
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
            label=reference_label,
        )
    if vertical_reference is not None:
        ax.axvline(
            vertical_reference,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
            label=vertical_reference_label,
        )
    if horizontal_reference is not None:
        ax.axhline(
            horizontal_reference,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
            label=horizontal_reference_label,
        )
    if log_x and all(record[x_metric] > 0 for record in usable_records):
        ax.set_xscale("log")
    if log_y and all(record[metric] > 0 for record in usable_records):
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_title:
        _set_title(ax, title)
    ax.legend(
        fontsize=(
            MULTIRUN_LEGEND_FONTSIZE
            if show_title
            else PRESENTATION_LEGEND_FONTSIZE
        ),
        markerscale=PRESENTATION_LEGEND_MARKERSCALE if not show_title else 1.0,
    )
    fig.tight_layout()
    return fig
