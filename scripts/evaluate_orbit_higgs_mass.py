#!/usr/bin/env python
"""Evaluate truth-free or legacy truth-matched Higgs mass fidelity on ggHbb."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_explicit_env_file() -> None:
    """Match ``gabbro/train.py`` support for a user-selected env file."""
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
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import awkward as ak
import fastjet
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import vector
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.optimize import curve_fit
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from gabbro.data.orbit_transforms import KinematicTransform
from gabbro.data.orbit_parquet import SEQUENCE_SCHEMAS, OrbitParquetDataset
from gabbro.plotting.orbit import (
    HISTOGRAM_FILL_ALPHA,
    HISTOGRAM_LINEWIDTH,
    ORIGINAL_COLOR,
    PRESENTATION_LEGEND_FONTSIZE,
    RECONSTRUCTED_COLOR,
    multirun_color,
)
from gabbro.plotting.utils import set_mpl_style


vector.register_awkward()

HIGGS_MASS_PLOT_BINS = np.arange(40.0, 205.0, 5.0)


TRUTH_COLUMNS = [
    "Gen_Part_PT",
    "Gen_Part_Eta",
    "Gen_Part_Phi",
    "Gen_Part_Mass",
    "Gen_Part_PID",
    "Gen_Part_D1",
    "Gen_Part_D2",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="One tokenizer run (legacy single-run mode).")
    source.add_argument(
        "--run",
        nargs="+",
        action="append",
        metavar="RUN_SPEC",
        help="Repeatable comparison entry: --run RUN_DIR or --run RUN_DIR LABEL.",
    )
    parser.add_argument("--gghbb-test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap-replicas", type=int, default=200)
    parser.add_argument(
        "--candidate-mode",
        choices=("leading_pt", "hungarian", "truth"),
        default="leading_pt",
        help="Candidate definition; the first two modes never access generator truth.",
    )
    parser.add_argument(
        "--max-abs-eta",
        type=float,
        help="Optional post-clustering |eta| acceptance, applied to both representations.",
    )
    parser.add_argument(
        "--max-match-dr",
        type=float,
        help="Optional Hungarian-match rejection threshold (hungarian/truth modes only).",
    )
    parser.add_argument(
        "--apply-current-cuts",
        action="store_true",
        help="Enable the applicable standard cuts: |eta| < 2.5 and match delta-R < 0.2.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="orbit-tokenizer")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-run-id",
        help=(
            "Resume an existing W&B run and log the Higgs outputs under its "
            "downstream namespace. Intended for attaching evaluation plots to "
            "the originating tokenizer training run."
        ),
    )
    parser.add_argument("--no-wandb", action="store_true", help="Keep results local; do not sync W&B.")
    return parser.parse_args()


def parse_named_runs(run_specs):
    """Parse collector-compatible ``--run RUN_DIR [LABEL]`` arguments."""
    runs = []
    for spec in run_specs:
        if len(spec) not in (1, 2):
            raise ValueError("Each --run entry must be '--run RUN_DIR' or '--run RUN_DIR LABEL'.")
        run_dir = Path(spec[0]).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        runs.append((run_dir, spec[1] if len(spec) == 2 else run_dir.name))
    labels = [label for _, label in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Each multi-run label must be unique.")
    return runs


def run_specs(args):
    if args.run_dir is not None:
        return [(args.run_dir.resolve(), args.run_dir.resolve().name)], False
    return parse_named_runs(args.run), True


def output_component(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not value:
        raise ValueError(f"Run label cannot be converted into an output directory: {label!r}")
    return value


def load_model(run_dir: Path, device):
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        config_path = run_dir / ".hydra" / "config.yaml"
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(f"No resolved or Hydra config found under {run_dir}")
    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
        model.load_state_dict(state["state_dict"], strict=True)
        checkpoint_path = checkpoint
    elif not hasattr(model, "_fit_faiss"):
        raise FileNotFoundError(checkpoint)
    else:
        # Test-only baselines (for example FAISS k-means) fit at evaluation
        # time and deliberately do not create a training checkpoint.
        checkpoint_path = None
    return model.to(device).eval(), cfg, checkpoint_path


def checkpoint_data_settings(cfg) -> tuple[str, int, str | None, float]:
    sequence_type = str(cfg.data.get("sequence_type", "particle"))
    if sequence_type not in SEQUENCE_SCHEMAS:
        raise ValueError(f"Unsupported checkpoint sequence type: {sequence_type!r}")
    configured_length = cfg.data.get("max_sequence_length")
    max_sequence_length = (
        int(configured_length)
        if configured_length is not None
        else int(SEQUENCE_SCHEMAS[sequence_type]["max_sequence_length"])
    )
    schema = SEQUENCE_SCHEMAS[sequence_type]
    configured_mask_column = cfg.data.get("mask_column")
    mask_column = (
        schema["mask_column"]
        if configured_mask_column is None
        else str(configured_mask_column)
    )
    configured_mask_minimum = cfg.data.get("mask_min_value")
    mask_min_value = (
        float(schema["mask_min_value"])
        if configured_mask_minimum is None
        else float(configured_mask_minimum)
    )
    return sequence_type, max_sequence_length, mask_column, mask_min_value


def checkpoint_energy_settings(cfg) -> tuple[bool, float]:
    """Recover optional energy-feature settings from the training data config."""
    return bool(cfg.data.get("include_energy", False)), float(
        cfg.data.get("energy_shift", 2.5)
    )


def decode_batch(model, batch, device, kinematics: KinematicTransform):
    features = batch["part_features"].to(device)
    mask = batch["part_mask"].to(device)
    pid = batch.get("part_pid")
    if pid is not None:
        pid = pid.to(device)
    with torch.inference_mode():
        if hasattr(model, "_quantize_batch"):
            reconstructed, _, _ = model._quantize_batch(features, mask, pid)
        else:
            reconstructed, _ = model.forward(features, mask, pid_particle=pid)
    return kinematics.to_physical(reconstructed.detach().cpu().numpy())


def prepare_test_time_baseline(model, cfg, output_dir: Path):
    if not (hasattr(model, "_fit_faiss") or getattr(model, "binning", None) == "learned"):
        return
    data_module = instantiate(cfg.data)
    model._trainer = SimpleNamespace(
        datamodule=data_module,
        is_global_zero=True,
        default_root_dir=str(output_dir),
        loggers=[],
    )
    if hasattr(model, "save_codebook"):
        model.save_codebook = False
    if hasattr(model, "upload_codebook_to_wandb"):
        model.upload_codebook_to_wandb = False
    if getattr(model, "binning", None) == "learned":
        model._fit_learned_bins()
    if hasattr(model, "_fit_faiss"):
        model._fit_faiss()


class TruthReader:
    """Read truth rows while caching only the currently used parquet row group."""

    def __init__(self):
        self._layouts = {}
        self._cache_key = None
        self._cache = None

    def row(self, path: str, row_index: int):
        if path not in self._layouts:
            parquet = pq.ParquetFile(path)
            offsets = [0]
            for index in range(parquet.num_row_groups):
                offsets.append(offsets[-1] + parquet.metadata.row_group(index).num_rows)
            self._layouts[path] = (parquet, offsets)
        parquet, offsets = self._layouts[path]
        row_group = bisect.bisect_right(offsets, row_index) - 1
        cache_key = (path, row_group)
        if cache_key != self._cache_key:
            self._cache = ak.from_arrow(parquet.read_row_group(row_group, columns=TRUTH_COLUMNS))
            self._cache_key = cache_key
        return self._cache[row_index - offsets[row_group]]


def decaying_higgs(truth):
    pids = np.asarray(ak.to_numpy(truth.Gen_Part_PID), dtype=np.int64)
    d1s = np.asarray(ak.to_numpy(truth.Gen_Part_D1), dtype=np.int64)
    d2s = np.asarray(ak.to_numpy(truth.Gen_Part_D2), dtype=np.int64)
    for higgs_index in np.flatnonzero(np.abs(pids) == 25)[::-1]:
        d1, d2 = int(d1s[higgs_index]), int(d2s[higgs_index])
        if 0 <= d1 < len(pids) and 0 <= d2 < len(pids) and {int(pids[d1]), int(pids[d2])} == {5, -5}:
            def particle(index):
                return np.array(
                    [
                        float(truth.Gen_Part_Eta[index]),
                        float(truth.Gen_Part_Phi[index]),
                        float(truth.Gen_Part_PT[index]),
                        float(truth.Gen_Part_Mass[index]),
                    ]
                )

            return particle(higgs_index), particle(d1), particle(d2)
    return None


def delta_r(eta_a, phi_a, eta_b, phi_b):
    delta_phi = np.remainder(phi_a - phi_b + np.pi, 2 * np.pi) - np.pi
    return np.hypot(eta_a - eta_b, delta_phi)


def cluster_jets(particles: np.ndarray, radius: float, min_pt: float):
    if not len(particles):
        return np.empty((0, 4))
    momenta = ak.zip(
        {
            "pt": [particles[:, 2]],
            "eta": [particles[:, 0]],
            "phi": [particles[:, 1]],
            "mass": [np.zeros(len(particles))],
        },
        with_name="Momentum4D",
    )
    jets = fastjet.ClusterSequence(
        momenta, fastjet.JetDefinition(fastjet.antikt_algorithm, radius)
    ).inclusive_jets(min_pt=min_pt)[0]
    if not len(jets):
        return np.empty((0, 4))
    values = np.stack(
        [ak.to_numpy(jets.eta), ak.to_numpy(jets.phi), ak.to_numpy(jets.pt), ak.to_numpy(jets.mass)],
        axis=-1,
    )
    return values[np.argsort(values[:, 2])[::-1]]


def accepted_objects(objects: np.ndarray, max_abs_eta: float | None):
    """Apply an optional post-reconstruction eta acceptance."""
    if max_abs_eta is None:
        return objects
    return objects[np.abs(objects[:, 0]) < max_abs_eta]


def resolved_jets(particles, max_abs_eta=None):
    """Return the accepted anti-kt R=0.4 jets used by the resolved benchmark."""
    return accepted_objects(
        cluster_jets(particles, radius=0.4, min_pt=30.0), max_abs_eta
    )


def combine_mass(first, second):
    def components(jet):
        eta, phi, pt, mass = jet
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        return np.array([energy, px, py, pz])

    total = components(first) + components(second)
    return float(np.sqrt(max(total[0] ** 2 - np.dot(total[1:], total[1:]), 0.0)))


def resolved_candidate(
    particles,
    b,
    bbar,
    max_abs_eta=2.5,
    max_match_dr=0.2,
    jets=None,
):
    """Legacy candidate formed by matching AK4 jets to truth b daughters."""
    jets = resolved_jets(particles, max_abs_eta) if jets is None else jets
    if len(jets) < 2:
        return None
    distances = np.array(
        [[delta_r(parton[0], parton[1], jet[0], jet[1]) for jet in jets] for parton in (b, bbar)]
    )
    parton_indices, jet_indices = linear_sum_assignment(distances)
    if len(jet_indices) != 2:
        return None
    if max_match_dr is not None and np.any(
        distances[parton_indices, jet_indices] >= max_match_dr
    ):
        return None
    return combine_mass(jets[jet_indices[0]], jets[jet_indices[1]])


def leading_higgs_candidate(particles, max_abs_eta=None):
    """Build a resolved candidate from the two leading reconstructed AK4 jets."""
    jets = resolved_jets(particles, max_abs_eta)
    return leading_higgs_candidate_from_jets(jets)


def leading_higgs_candidate_from_jets(jets):
    """Build the leading-dijet candidate from an already-clustered jet collection."""
    if len(jets) < 2:
        return None
    return {
        "mass": combine_mass(jets[0], jets[1]),
        "jets": jets[:2],
        "indices": np.array([0, 1], dtype=np.int64),
        "match_dr": np.array([np.nan, np.nan]),
    }


def cross_matched_higgs_candidates(
    original_particles,
    decoded_particles,
    max_abs_eta=None,
    max_match_dr=None,
):
    """Anchor on the original leading jets and match them to decoded AK4 jets."""
    original_jets = resolved_jets(original_particles, max_abs_eta)
    decoded_jets = resolved_jets(decoded_particles, max_abs_eta)
    return cross_matched_higgs_candidates_from_jets(
        original_jets, decoded_jets, max_match_dr=max_match_dr
    )


def cross_matched_higgs_candidates_from_jets(
    original_jets,
    decoded_jets,
    max_match_dr=None,
):
    """Match decoded jets to original leading jets using cached jet collections."""
    original = leading_higgs_candidate_from_jets(original_jets)
    if original is None or len(decoded_jets) < 2:
        return original, None
    distances = np.array(
        [
            [delta_r(reference[0], reference[1], jet[0], jet[1]) for jet in decoded_jets]
            for reference in original["jets"]
        ]
    )
    reference_indices, decoded_indices = linear_sum_assignment(distances)
    if len(decoded_indices) != 2:
        return original, None
    ordered_indices = np.empty(2, dtype=np.int64)
    ordered_indices[reference_indices] = decoded_indices
    matched_dr = distances[np.arange(2), ordered_indices]
    if max_match_dr is not None and np.any(matched_dr >= max_match_dr):
        return original, None
    matched_jets = decoded_jets[ordered_indices]
    decoded = {
        "mass": combine_mass(matched_jets[0], matched_jets[1]),
        "jets": matched_jets,
        "indices": ordered_indices,
        "match_dr": matched_dr,
    }
    return original, decoded


def boosted_candidate(particles, higgs, b, bbar):
    jets = accepted_objects(cluster_jets(particles, radius=0.8, min_pt=250.0), 2.5)
    if not len(jets):
        return None
    distances = np.array([delta_r(higgs[0], higgs[1], jet[0], jet[1]) for jet in jets])
    jet = jets[int(np.argmin(distances))]
    if np.min(distances) >= 0.4:
        return None
    if delta_r(b[0], b[1], jet[0], jet[1]) >= 0.8:
        return None
    if delta_r(bbar[0], bbar[1], jet[0], jet[1]) >= 0.8:
        return None
    return float(jet[3])


def double_crystal_ball(x, norm, mean, sigma, alpha_l, n_l, alpha_r, n_r):
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


def fit_mass(values):
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
        return {
            "success": True,
            "events": int(len(values)),
            "mean": float(parameters[1]),
            "sigma": float(parameters[2]),
            "parameters": list(map(float, parameters)),
        }
    except (RuntimeError, ValueError):
        return {"success": False, "events": int(len(values)), "mean": None, "sigma": None}


def bootstrap_fit(values, replicas):
    if len(values) < 50 or replicas < 1:
        return {
            "successful_replicas": 0,
            "mean_uncertainty": None,
            "sigma_uncertainty": None,
        }
    rng = np.random.default_rng(12345)
    fits = [fit_mass(rng.choice(values, size=len(values), replace=True)) for _ in range(replicas)]
    means = [fit["mean"] for fit in fits if fit["success"]]
    sigmas = [fit["sigma"] for fit in fits if fit["success"]]
    return {
        "successful_replicas": len(means),
        "mean_uncertainty": float(np.std(means, ddof=1)) if len(means) > 1 else None,
        "sigma_uncertainty": float(np.std(sigmas, ddof=1)) if len(sigmas) > 1 else None,
    }


def distribution_moments(values):
    """Return empirical moments for the finite mass values shown in a histogram."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "events": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "std": float(np.std(values)) if len(values) else None,
    }


def distribution_label(label, values):
    moments = distribution_moments(values)
    if moments["mean"] is None:
        return label
    return f"{label} ($\\mu$={moments['mean']:.1f}, $\\sigma$={moments['std']:.1f} GeV)"


def fitted_distribution_label(label, values):
    fit = fit_mass(values)
    if not fit["success"]:
        return label
    return f"{label} ($\\mu$={fit['mean']:.1f}, $\\sigma$={fit['sigma']:.1f} GeV)"


def summarize_topology(rows, topology, replicas):
    original = np.array([row[f"{topology}_original"] for row in rows], dtype=float)
    decoded = np.array([row[f"{topology}_decoded"] for row in rows], dtype=float)
    common = np.isfinite(original) & np.isfinite(decoded)
    original_common, decoded_common = original[common], decoded[common]
    original_fit, decoded_fit = fit_mass(original_common), fit_mass(decoded_common)
    if len(original_common):
        paired_mean = float(np.mean(decoded_common - original_common))
        paired_std = float(np.std(decoded_common - original_common))
        original_width = float(np.diff(np.quantile(original_common, [0.16, 0.84]))[0] / 2)
        decoded_width = float(np.diff(np.quantile(decoded_common, [0.16, 0.84]))[0] / 2)
    else:
        paired_mean = paired_std = original_width = decoded_width = None
    summary = {
        "events": len(rows),
        "original_efficiency": float(np.mean(np.isfinite(original))),
        "decoded_efficiency": float(np.mean(np.isfinite(decoded))),
        "common_events": int(common.sum()),
        "original_distribution": distribution_moments(original_common),
        "decoded_distribution": distribution_moments(decoded_common),
        "original_fit": original_fit | bootstrap_fit(original_common, replicas),
        "decoded_fit": decoded_fit | bootstrap_fit(decoded_common, replicas),
        "paired_mass_difference_mean": paired_mean,
        "paired_mass_difference_std": paired_std,
        "original_central68_half_width": original_width,
        "decoded_central68_half_width": decoded_width,
    }
    if original_fit["success"] and decoded_fit["success"]:
        summary["fit_peak_shift"] = decoded_fit["mean"] - original_fit["mean"]
        summary["fit_width_ratio"] = decoded_fit["sigma"] / original_fit["sigma"]
    else:
        summary["fit_peak_shift"] = None
        summary["fit_width_ratio"] = None
    if rows and "truth_higgs_pt" in rows[0]:
        pt = np.asarray([row["truth_higgs_pt"] for row in rows], dtype=float)
        summary["truth_higgs_pt_bins"] = {}
        for low, high in zip((0, 250, 350, 500, 750), (250, 350, 500, 750, np.inf)):
            selected = (pt >= low) & (pt < high)
            selected_common = selected & common
            name = f"{low:g}_{high:g}" if np.isfinite(high) else f"{low:g}_inf"
            summary["truth_higgs_pt_bins"][name] = {
                "events": int(selected.sum()),
                "original_efficiency": float(np.mean(np.isfinite(original[selected])))
                if np.any(selected)
                else None,
                "decoded_efficiency": float(np.mean(np.isfinite(decoded[selected])))
                if np.any(selected)
                else None,
                "common_events": int(selected_common.sum()),
                "paired_mass_difference_mean": float(
                    np.mean(decoded[selected_common] - original[selected_common])
                )
                if np.any(selected_common)
                else None,
            }
    return summary


def mass_distribution_figure(
    original,
    decoded,
    topology="resolved",
    candidate_mode="truth",
    presentation=False,
    bins=None,
):
    """Render paired Higgs candidates with optional presentation styling."""
    set_mpl_style()
    original = np.asarray(original, dtype=float)
    decoded = np.asarray(decoded, dtype=float)
    common = np.isfinite(original) & np.isfinite(decoded)
    original = original[common]
    decoded = decoded[common]
    figure, axis = plt.subplots(figsize=(7, 5))
    bins = HIGGS_MASS_PLOT_BINS if bins is None else np.asarray(bins, dtype=float)
    label_fn = fitted_distribution_label if presentation else distribution_label
    axis.hist(
        original,
        bins=bins,
        histtype="stepfilled" if presentation else "step",
        density=True,
        color=ORIGINAL_COLOR if presentation else None,
        alpha=HISTOGRAM_FILL_ALPHA if presentation else None,
        linewidth=HISTOGRAM_LINEWIDTH,
        label=label_fn("Original", original),
    )
    axis.hist(
        decoded,
        bins=bins,
        histtype="step",
        density=True,
        color=RECONSTRUCTED_COLOR if presentation else None,
        linewidth=HISTOGRAM_LINEWIDTH,
        label=label_fn("Decoded", decoded),
    )
    axis.set(xlabel="Higgs candidate mass [GeV]", ylabel="Normalized events")
    if not presentation:
        axis.set_title(f"{topology} — {candidate_mode.replace('_', ' ')}")
    axis.legend(
        fontsize=PRESENTATION_LEGEND_FONTSIZE if presentation else 8,
    )
    figure.tight_layout()
    return figure


def plot_masses(rows, topology, output_dir, candidate_mode="truth"):
    original = np.array([row[f"{topology}_original"] for row in rows], dtype=float)
    decoded = np.array([row[f"{topology}_decoded"] for row in rows], dtype=float)
    common = np.isfinite(original) & np.isfinite(decoded)
    bins = HIGGS_MASS_PLOT_BINS
    figure = mass_distribution_figure(original, decoded, topology, candidate_mode)
    figure.savefig(output_dir / f"higgs_mass_{topology}.png", dpi=180)
    plt.close(figure)
    np.savez_compressed(
        output_dir / f"higgs_mass_{topology}_histograms.npz",
        bins=bins,
        original=original[common],
        decoded=decoded[common],
    )


def plot_event_observables(rows, output_dir):
    """Plot event-level particle and jet diagnostics beside the mass benchmark."""
    set_mpl_style()
    specifications = (
        (
            "leading_particle_pt",
            r"Leading particle $p_{\mathrm{T}}$ [GeV]",
            output_dir / "leading_particle_pt.png",
        ),
        ("n_jets", "Number of AK4 jets", output_dir / "n_jets.png"),
    )
    for name, xlabel, output_path in specifications:
        original = np.asarray([row[f"{name}_original"] for row in rows], dtype=float)
        decoded = np.asarray([row[f"{name}_decoded"] for row in rows], dtype=float)
        finite = np.concatenate((original[np.isfinite(original)], decoded[np.isfinite(decoded)]))
        if name == "n_jets":
            upper = int(np.max(finite)) if len(finite) else 0
            bins = np.arange(-0.5, upper + 1.5, 1.0)
        else:
            upper = max(float(np.max(finite)) if len(finite) else 1.0, 1.0)
            bins = np.linspace(0.0, upper, 61)
        figure, axis = plt.subplots(figsize=(7, 5))
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
        axis.set(xlabel=xlabel, ylabel="Normalized events")
        axis.set_yscale("log")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        np.savez_compressed(
            output_dir / f"{name}_histograms.npz",
            bins=bins,
            original=original,
            decoded=decoded,
        )


def plot_multirun_masses(results, topology, output_dir, candidate_mode="truth"):
    """Overlay each decoded distribution and one common original reference."""
    set_mpl_style()
    figure, axis = plt.subplots(figsize=(7, 5))
    bins = HIGGS_MASS_PLOT_BINS
    first_label, first_rows, _ = results[0]
    original = np.asarray([row[f"{topology}_original"] for row in first_rows], dtype=float)
    original = original[np.isfinite(original)]
    plot_data = {"bins": bins, "original": original, "original_label": np.asarray(first_label)}
    if len(original):
        axis.hist(
            original,
            bins=bins,
            histtype="step",
            density=True,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=distribution_label(f"Original ({first_label})", original),
        )
    for index, (label, rows, _) in enumerate(results):
        decoded = np.asarray([row[f"{topology}_decoded"] for row in rows], dtype=float)
        decoded = decoded[np.isfinite(decoded)]
        if len(decoded):
            axis.hist(
                decoded,
                bins=bins,
                histtype="step",
                density=True,
                linewidth=1.7,
                color=multirun_color(label),
                label=distribution_label(label, decoded),
            )
        plot_data[f"decoded_{index}"] = decoded
        plot_data[f"decoded_{index}_label"] = np.asarray(label)
    axis.set(
        xlabel="Higgs candidate mass [GeV]",
        ylabel="Normalized events",
        title=f"{topology} — {candidate_mode.replace('_', ' ')} decoded comparison",
    )
    axis.legend(prop={"size": 8})
    figure.tight_layout()
    figure.savefig(output_dir / f"higgs_mass_{topology}_multirun.png", dpi=180)
    plt.close(figure)
    np.savez_compressed(output_dir / f"higgs_mass_{topology}_multirun_histograms.npz", **plot_data)


def upload_to_wandb(args, output_dir: Path, multirun: bool, results):
    """Sync rendered plots and compact replotting inputs as one W&B artifact."""
    if args.no_wandb:
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B synchronization requested, but wandb is not installed.") from exc

    evaluation_config = {
        "events": args.events,
        "bootstrap_replicas": args.bootstrap_replicas,
        "candidate_mode": args.candidate_mode,
        "max_abs_eta": args.max_abs_eta,
        "max_match_dr": args.max_match_dr,
    }
    if args.wandb_run_id:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            resume="must",
        )
        metric_prefix = "downstream/higgs_mass"
        for key, value in evaluation_config.items():
            run.summary[f"{metric_prefix}/config/{key}"] = value
    else:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"higgs-mass-{output_dir.name}",
            group=args.wandb_group or "orbit-higgs-mass",
            entity=args.wandb_entity,
            job_type="multirun-comparison" if multirun else "evaluation",
            config=evaluation_config,
        )
        metric_prefix = "higgs_mass"
    try:
        images = {
            f"{metric_prefix}/{path.stem}": wandb.Image(str(path))
            for path in sorted(output_dir.rglob("*.png"))
        }
        if images:
            run.log(images)
        scalar_metrics = {}
        for label, _, summary in results:
            run_label = output_component(label)
            run_component = "" if args.wandb_run_id else f"/{run_label}"
            for topology in ("resolved", "boosted"):
                if topology not in summary:
                    continue
                for representation in ("original", "decoded"):
                    moments = summary[topology][f"{representation}_distribution"]
                    for statistic in ("mean", "std"):
                        value = moments[statistic]
                        if value is not None:
                            scalar_metrics[
                                f"{metric_prefix}/mass_moments{run_component}/{topology}/"
                                f"{representation}_{statistic}"
                            ] = value
                    fit = summary[topology][f"{representation}_fit"]
                    for statistic in ("mean", "sigma"):
                        value = fit[statistic]
                        if value is not None:
                            scalar_metrics[
                                f"{metric_prefix}/mass_fit{run_component}/{topology}/"
                                f"{representation}_{statistic}"
                            ] = value
        if scalar_metrics:
            run.log(scalar_metrics)
        artifact = wandb.Artifact(
            name=f"higgs-mass-{output_component(output_dir.name)}-{run.id}",
            type="orbit-downstream-evaluation",
            metadata={"multirun": multirun, "events_per_run": args.events},
        )
        for path in sorted(output_dir.rglob("*.npz")) + sorted(output_dir.rglob("*.json")):
            artifact.add_file(str(path), name=str(path.relative_to(output_dir)))
        run.log_artifact(artifact)
    finally:
        wandb.finish()


def evaluate_run(args, run_dir: Path, output_dir: Path):
    device = torch.device(args.device)
    model, cfg, checkpoint = load_model(run_dir, device)
    sequence_type, max_sequence_length, mask_column, mask_min_value = (
        checkpoint_data_settings(cfg)
    )
    include_energy, energy_shift = checkpoint_energy_settings(cfg)
    kinematics = KinematicTransform.from_data_config(cfg.data)
    batch_size = args.batch_size or (16 if max_sequence_length > 128 else 256)
    prepare_test_time_baseline(model, cfg, output_dir)
    dataset = OrbitParquetDataset(
        args.gghbb_test_manifest,
        sequence_type=sequence_type,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        mask_column=mask_column,
        mask_min_value=mask_min_value,
        return_raw_features=True,
        return_event_metadata=True,
        pid_cfg=cfg.get("pid"),
        include_energy=include_energy,
        energy_shift=energy_shift,
        **kinematics.as_kwargs(),
    )
    truth_reader = TruthReader() if args.candidate_mode == "truth" else None
    rows = []
    with tqdm(total=args.events, desc="Evaluating ggHbb", unit="event") as progress:
        for batch in dataset:
            decoded = decode_batch(model, batch, device, kinematics)
            mask = batch["part_mask"].numpy().astype(bool)
            for index in range(mask.shape[0]):
                source_file = str(batch["source_files"][index])
                source_row = int(batch["source_rows"][index])
                raw = np.asarray(ak.to_numpy(batch["raw_part_features"][index]), dtype=float)
                original = raw[:, [1, 2, 0]]
                decoded_event = decoded[index, mask[index]]
                original_jets = resolved_jets(original, args.max_abs_eta)
                decoded_jets = resolved_jets(decoded_event, args.max_abs_eta)
                row = {
                    "source_file": source_file,
                    "source_row": source_row,
                    "leading_particle_pt_original": float(np.max(original[:, 2]))
                    if len(original)
                    else float("nan"),
                    "leading_particle_pt_decoded": float(np.max(decoded_event[:, 2]))
                    if len(decoded_event)
                    else float("nan"),
                    "n_jets_original": int(len(original_jets)),
                    "n_jets_decoded": int(len(decoded_jets)),
                }
                if args.candidate_mode == "truth":
                    truth = decaying_higgs(truth_reader.row(source_file, source_row))
                    if truth is None:
                        continue
                    higgs, b, bbar = truth
                    row.update(
                        {
                            "truth_higgs_pt": float(higgs[2]),
                            "resolved_original": resolved_candidate(
                                original,
                                b,
                                bbar,
                                args.max_abs_eta,
                                args.max_match_dr,
                                jets=original_jets,
                            ),
                            "resolved_decoded": resolved_candidate(
                                decoded_event,
                                b,
                                bbar,
                                args.max_abs_eta,
                                args.max_match_dr,
                                jets=decoded_jets,
                            ),
                            "boosted_original": boosted_candidate(original, higgs, b, bbar),
                            "boosted_decoded": boosted_candidate(decoded_event, higgs, b, bbar),
                        }
                    )
                elif args.candidate_mode == "leading_pt":
                    original_candidate = leading_higgs_candidate_from_jets(original_jets)
                    decoded_candidate = leading_higgs_candidate_from_jets(decoded_jets)
                    row.update(
                        {
                            "resolved_original": original_candidate["mass"]
                            if original_candidate is not None
                            else None,
                            "resolved_decoded": decoded_candidate["mass"]
                            if decoded_candidate is not None
                            else None,
                            "decoded_jet0_match_dr": float("nan"),
                            "decoded_jet1_match_dr": float("nan"),
                        }
                    )
                else:
                    original_candidate, decoded_candidate = cross_matched_higgs_candidates_from_jets(
                        original_jets,
                        decoded_jets,
                        max_match_dr=args.max_match_dr,
                    )
                    row.update(
                        {
                            "resolved_original": original_candidate["mass"]
                            if original_candidate is not None
                            else None,
                            "resolved_decoded": decoded_candidate["mass"]
                            if decoded_candidate is not None
                            else None,
                            "decoded_jet0_match_dr": float(decoded_candidate["match_dr"][0])
                            if decoded_candidate is not None
                            else float("nan"),
                            "decoded_jet1_match_dr": float(decoded_candidate["match_dr"][1])
                            if decoded_candidate is not None
                            else float("nan"),
                        }
                    )
                rows.append(row)
                progress.update(1)
                if len(rows) == args.events:
                    break
            if len(rows) == args.events:
                break
    if len(rows) != args.events:
        qualifier = "truth-selected " if args.candidate_mode == "truth" else ""
        raise RuntimeError(f"Expected {args.events} {qualifier}events, got {len(rows)}")
    for row in rows:
        mass_keys = ["resolved_original", "resolved_decoded"]
        if args.candidate_mode == "truth":
            mass_keys += ["boosted_original", "boosted_decoded"]
        for key in mass_keys:
            if row[key] is None:
                row[key] = float("nan")
    with (output_dir / "higgs_candidates.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "events": args.events,
        "sequence_type": sequence_type,
        "max_sequence_length": max_sequence_length,
        "candidate_mode": args.candidate_mode,
        "particle_selection": {
            "mask_column": mask_column,
            "mask_min_value": mask_min_value,
            "weighted_pt": False,
        },
        "clustering": {
            "algorithm": "antikt",
            "resolved": {"radius": 0.4, "min_pt": 30.0},
        },
        "postprocessing": {
            "max_abs_eta": args.max_abs_eta,
            "max_match_dr": args.max_match_dr,
        },
        "candidate_selection": (
            "independent two leading-pT AK4 jets"
            if args.candidate_mode == "leading_pt"
            else "original leading AK4 jets matched to decoded AK4 jets by Hungarian delta-R"
            if args.candidate_mode == "hungarian"
            else "legacy independent Hungarian matching to generator b daughters"
        ),
        "resolved": summarize_topology(rows, "resolved", args.bootstrap_replicas),
    }
    if args.candidate_mode == "truth":
        summary["clustering"]["boosted"] = {
            "radius": 0.8,
            "min_pt": 250.0,
            "max_abs_eta": 2.5,
            "match_dr": 0.4,
        }
        summary["boosted"] = summarize_topology(rows, "boosted", args.bootstrap_replicas)
    (output_dir / "higgs_mass_metrics.json").write_text(json.dumps(summary, indent=2))
    plot_masses(rows, "resolved", output_dir, args.candidate_mode)
    plot_event_observables(rows, output_dir)
    if args.candidate_mode == "truth":
        plot_masses(rows, "boosted", output_dir, args.candidate_mode)
    return rows, summary


def main():
    args = parse_args()
    if args.no_wandb and args.wandb_run_id:
        raise ValueError("--wandb-run-id cannot be combined with --no-wandb")
    if args.events < 1:
        raise ValueError("--events must be positive")
    if args.apply_current_cuts:
        args.max_abs_eta = 2.5 if args.max_abs_eta is None else args.max_abs_eta
        if args.candidate_mode != "leading_pt":
            args.max_match_dr = 0.2 if args.max_match_dr is None else args.max_match_dr
    if args.max_abs_eta is not None and args.max_abs_eta <= 0:
        raise ValueError("--max-abs-eta must be positive")
    if args.max_match_dr is not None and args.max_match_dr <= 0:
        raise ValueError("--max-match-dr must be positive")
    if args.candidate_mode == "leading_pt" and args.max_match_dr is not None:
        raise ValueError("--max-match-dr only applies to hungarian or truth mode")
    runs, multirun = run_specs(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    used_components = set()
    for run_dir, label in runs:
        run_output_dir = args.output_dir if not multirun else args.output_dir / "runs" / output_component(label)
        if run_output_dir.name in used_components:
            raise ValueError("Run labels map to duplicate output directory names.")
        used_components.add(run_output_dir.name)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        rows, summary = evaluate_run(args, run_dir, run_output_dir)
        results.append((label, rows, summary))
        print(f"Wrote Higgs mass evaluation for {label!r} to {run_output_dir}")
    if multirun:
        topologies = ("resolved", "boosted") if args.candidate_mode == "truth" else ("resolved",)
        for topology in topologies:
            plot_multirun_masses(results, topology, args.output_dir, args.candidate_mode)
        comparison = {
            "gghbb_test_manifest": str(args.gghbb_test_manifest.resolve()),
            "events_per_run": args.events,
            "runs": [
                {"label": label, "run_dir": str(run_dir), "metrics": summary}
                for (run_dir, label), (_, _, summary) in zip(runs, results, strict=True)
            ],
        }
        (args.output_dir / "higgs_mass_multirun_metrics.json").write_text(
            json.dumps(comparison, indent=2)
        )
        print(f"Wrote Higgs multi-run comparison to {args.output_dir}")
    upload_to_wandb(args, args.output_dir, multirun, results)


if __name__ == "__main__":
    main()
