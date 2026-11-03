#!/usr/bin/env python
"""Evaluate truth-free or legacy truth-matched Z→μ⁺μ⁻ mass fidelity."""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_Z_TEST_MANIFEST = (
    Path(os.environ.get("ORBIT_MANIFEST_DIR", PROJECT_ROOT / "manifests" / "production_final"))
    / "ZZ_leptonic_test.txt"
)


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
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.optimize import curve_fit
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from gabbro.data.orbit_parquet import SEQUENCE_SCHEMAS, OrbitParquetDataset
from gabbro.plotting.orbit import multirun_color
from gabbro.plotting.utils import set_mpl_style


MUON_PID = 6
ANTIMUON_PID = 7
MUON_MASS_GEV = 0.1056583755
Z_MASS_GEV = 91.1876
MUON_MATCH_DR = 0.2
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
    source.add_argument(
        "--config",
        type=Path,
        help="Resolved Hydra config or exported W&B config for an explicit checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint paired with --config (for models without a local run directory).",
    )
    parser.add_argument(
        "--z-test-manifest",
        "--dyjets-test-manifest",
        dest="z_test_manifest",
        type=Path,
        default=DEFAULT_Z_TEST_MANIFEST,
        help=(
            "Input manifest (default: ZZ_leptonic_test.txt under ORBIT_MANIFEST_DIR). "
            "--dyjets-test-manifest remains available as a compatibility alias."
        ),
    )
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
        help="Optional muon |eta| acceptance, applied to both representations.",
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
    if args.config is not None:
        return [(args.config.resolve().parent, args.checkpoint.stem)], False
    return parse_named_runs(args.run), True


def output_component(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not value:
        raise ValueError(f"Run label cannot be converted into an output directory: {label!r}")
    return value


def load_config(config_path: Path):
    """Load either a native Hydra config or a W&B-exported config."""
    payload = yaml.safe_load(config_path.read_text())
    if isinstance(payload.get("data"), dict) and "value" in payload["data"]:
        unwrapped = {}
        for key, entry in payload.items():
            value = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
            if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    pass
            unwrapped[key] = value
        payload = unwrapped
    if "pid" not in payload and "pid_cfg" in payload:
        payload["pid"] = payload["pid_cfg"]
    return OmegaConf.create(payload)


def load_model(
    run_dir: Path,
    device: torch.device,
    *,
    explicit_config: Path | None = None,
    explicit_checkpoint: Path | None = None,
):
    config_path = explicit_config or run_dir / "config_resolved.yaml"
    if explicit_config is None and not config_path.is_file():
        config_path = run_dir / ".hydra" / "config.yaml"
    checkpoint = explicit_checkpoint or run_dir / "checkpoints" / "best.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(f"No resolved or Hydra config found under {run_dir}")
    cfg = load_config(config_path)
    if not bool(cfg.get("pid", {}).get("enabled", False)):
        raise ValueError("Z→μ⁺μ⁻ evaluation requires a PID-enabled checkpoint")
    model = instantiate(cfg.model)
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
        model.load_state_dict(state["state_dict"], strict=True)
        checkpoint_path = checkpoint
    elif not hasattr(model, "_fit_faiss"):
        raise FileNotFoundError(checkpoint)
    else:
        # Test-only baselines fit their representation on demand.
        checkpoint_path = None
    return model.to(device).eval(), cfg, checkpoint_path


def checkpoint_data_settings(cfg) -> tuple[str, int, str | None, float]:
    sequence_type = str(cfg.data.get("sequence_type", "particle"))
    if sequence_type not in SEQUENCE_SCHEMAS:
        raise ValueError(f"Unsupported checkpoint sequence type: {sequence_type!r}")
    schema = SEQUENCE_SCHEMAS[sequence_type]
    max_sequence_length = int(
        cfg.data.get("max_sequence_length") or schema["max_sequence_length"]
    )
    mask_column = cfg.data.get("mask_column")
    if mask_column is None:
        mask_column = schema["mask_column"]
    mask_min_value = cfg.data.get("mask_min_value")
    if mask_min_value is None:
        mask_min_value = schema["mask_min_value"]
    return sequence_type, max_sequence_length, mask_column, float(mask_min_value)


def checkpoint_energy_settings(cfg) -> tuple[bool, float]:
    """Recover optional energy-feature settings from the training data config."""
    return bool(cfg.data.get("include_energy", False)), float(
        cfg.data.get("energy_shift", 2.5)
    )


def decode_batch(model, batch, device: torch.device):
    features = batch["part_features"].to(device)
    mask = batch["part_mask"].to(device)
    pid = batch["part_pid"].to(device)
    with torch.inference_mode():
        if hasattr(model, "_quantize_batch"):
            reconstructed, _, pid_logits = model._quantize_batch(features, mask, pid)
        else:
            reconstructed, outputs = model.forward(features, mask, pid_particle=pid)
            pid_logits = outputs.get("pid_logits")
    if pid_logits is None:
        raise ValueError("PID-enabled checkpoint did not return decoded PID logits")
    values = reconstructed.detach().cpu().numpy()
    physical = np.stack(
        [
            values[..., 0] * 3.0,
            np.arctan2(values[..., 2], values[..., 1]),
            np.clip(np.exp(values[..., 3] + 1.8) - 1e-8, 0.0, None),
        ],
        axis=-1,
    )
    return physical, pid_logits.argmax(dim=-1).detach().cpu().numpy()


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


def decaying_z_to_mumu(truth):
    """Return the last generator Z and direct μ⁻/μ⁺ daughters, if present."""
    pids = np.asarray(ak.to_numpy(truth.Gen_Part_PID), dtype=np.int64)
    d1s = np.asarray(ak.to_numpy(truth.Gen_Part_D1), dtype=np.int64)
    d2s = np.asarray(ak.to_numpy(truth.Gen_Part_D2), dtype=np.int64)
    for z_index in np.flatnonzero(pids == 23)[::-1]:
        d1, d2 = int(d1s[z_index]), int(d2s[z_index])
        if 0 <= d1 < len(pids) and 0 <= d2 < len(pids) and {int(pids[d1]), int(pids[d2])} == {13, -13}:
            def particle(index):
                return np.array(
                    [
                        float(truth.Gen_Part_Eta[index]),
                        float(truth.Gen_Part_Phi[index]),
                        float(truth.Gen_Part_PT[index]),
                        float(truth.Gen_Part_Mass[index]),
                    ]
                )

            muon_index = d1 if int(pids[d1]) == 13 else d2
            antimuon_index = d1 if int(pids[d1]) == -13 else d2
            return {
                "z": particle(z_index),
                "muon": particle(muon_index),
                "antimuon": particle(antimuon_index),
            }
    return None


def invariant_mass(first: np.ndarray, second: np.ndarray, mass: float = MUON_MASS_GEV) -> float:
    """Combine [eta, phi, pt] particles using a fixed on-shell mass."""
    def components(particle):
        eta, phi, pt = particle
        px, py = pt * np.cos(phi), pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        return np.array([energy, px, py, pz])

    total = components(first) + components(second)
    return float(np.sqrt(max(total[0] ** 2 - np.dot(total[1:], total[1:]), 0.0)))


def delta_r(eta_a, phi_a, eta_b, phi_b):
    delta_phi = np.remainder(phi_a - phi_b + np.pi, 2 * np.pi) - np.pi
    return np.hypot(eta_a - eta_b, delta_phi)


def truth_matched_dimuon_candidate(
    particles: np.ndarray,
    pid: np.ndarray,
    truth_muon: np.ndarray,
    truth_antimuon: np.ndarray,
    match_dr: float = MUON_MATCH_DR,
    max_abs_eta: float | None = None,
):
    """Build a PID-constrained dimuon candidate matched to direct truth daughters."""
    particles = np.asarray(particles, dtype=float)
    pid = np.asarray(pid, dtype=np.int64)
    if len(particles) != len(pid):
        raise ValueError("particles and pid must have the same length")
    eligible_indices = np.arange(len(particles))
    if max_abs_eta is not None:
        eligible_indices = eligible_indices[np.abs(particles[:, 0]) < max_abs_eta]
    if len(eligible_indices) < 2:
        return None
    eligible_particles = particles[eligible_indices]
    distances = np.array(
        [
            [delta_r(truth[0], truth[1], particle[0], particle[1]) for particle in eligible_particles]
            for truth in (truth_muon, truth_antimuon)
        ]
    )
    truth_indices, particle_indices = linear_sum_assignment(distances)
    if len(particle_indices) != 2:
        return None
    matched_indices = dict(zip(truth_indices, particle_indices, strict=True))
    muon_local_index, antimuon_local_index = matched_indices[0], matched_indices[1]
    muon_dr = distances[0, muon_local_index]
    antimuon_dr = distances[1, antimuon_local_index]
    if match_dr is not None and (muon_dr >= match_dr or antimuon_dr >= match_dr):
        return None
    muon_index = int(eligible_indices[muon_local_index])
    antimuon_index = int(eligible_indices[antimuon_local_index])
    if pid[muon_index] != MUON_PID or pid[antimuon_index] != ANTIMUON_PID:
        return None
    return {
        "mass": invariant_mass(particles[muon_index], particles[antimuon_index]),
        "muon": particles[muon_index],
        "antimuon": particles[antimuon_index],
        "muon_index": muon_index,
        "antimuon_index": antimuon_index,
        "muon_dr": float(muon_dr),
        "antimuon_dr": float(antimuon_dr),
        "muon_pid": int(pid[muon_index]),
        "antimuon_pid": int(pid[antimuon_index]),
    }


def leading_dimuon_candidate(particles, pid, max_abs_eta=None):
    """Select the leading reconstructed muon and antimuon by PID."""
    particles = np.asarray(particles, dtype=float)
    pid = np.asarray(pid, dtype=np.int64)
    if len(particles) != len(pid):
        raise ValueError("particles and pid must have the same length")
    eligible = np.ones(len(particles), dtype=bool)
    if max_abs_eta is not None:
        eligible &= np.abs(particles[:, 0]) < max_abs_eta
    muon_indices = np.flatnonzero(eligible & (pid == MUON_PID))
    antimuon_indices = np.flatnonzero(eligible & (pid == ANTIMUON_PID))
    if not len(muon_indices) or not len(antimuon_indices):
        return None
    muon_index = int(muon_indices[np.argmax(particles[muon_indices, 2])])
    antimuon_index = int(antimuon_indices[np.argmax(particles[antimuon_indices, 2])])
    return {
        "mass": invariant_mass(particles[muon_index], particles[antimuon_index]),
        "muon": particles[muon_index],
        "antimuon": particles[antimuon_index],
        "muon_index": muon_index,
        "antimuon_index": antimuon_index,
        "muon_dr": float("nan"),
        "antimuon_dr": float("nan"),
        "muon_pid": int(pid[muon_index]),
        "antimuon_pid": int(pid[antimuon_index]),
    }


def cross_matched_dimuon_candidates(
    original_particles,
    original_pid,
    decoded_particles,
    decoded_pid,
    max_abs_eta=None,
    max_match_dr=None,
):
    """Anchor on original leading muons and Hungarian-match compatible decoded PIDs."""
    original = leading_dimuon_candidate(original_particles, original_pid, max_abs_eta)
    if original is None:
        return None, None
    decoded_particles = np.asarray(decoded_particles, dtype=float)
    decoded_pid = np.asarray(decoded_pid, dtype=np.int64)
    if len(decoded_particles) != len(decoded_pid):
        raise ValueError("particles and pid must have the same length")
    eligible = np.isin(decoded_pid, (MUON_PID, ANTIMUON_PID))
    if max_abs_eta is not None:
        eligible &= np.abs(decoded_particles[:, 0]) < max_abs_eta
    candidate_indices = np.flatnonzero(eligible)
    if len(candidate_indices) < 2:
        return original, None
    candidates = decoded_particles[candidate_indices]
    candidate_pid = decoded_pid[candidate_indices]
    references = (original["muon"], original["antimuon"])
    expected_pid = (MUON_PID, ANTIMUON_PID)
    distances = np.full((2, len(candidates)), 1e9, dtype=float)
    for reference_index, (reference, required_pid) in enumerate(
        zip(references, expected_pid, strict=True)
    ):
        compatible = np.flatnonzero(candidate_pid == required_pid)
        distances[reference_index, compatible] = [
            delta_r(reference[0], reference[1], candidates[index, 0], candidates[index, 1])
            for index in compatible
        ]
    reference_indices, local_indices = linear_sum_assignment(distances)
    if len(local_indices) != 2:
        return original, None
    ordered_local = np.empty(2, dtype=np.int64)
    ordered_local[reference_indices] = local_indices
    matched_dr = distances[np.arange(2), ordered_local]
    if np.any(matched_dr >= 1e9):
        return original, None
    if max_match_dr is not None and np.any(matched_dr >= max_match_dr):
        return original, None
    matched_indices = candidate_indices[ordered_local]
    muon_index, antimuon_index = map(int, matched_indices)
    decoded = {
        "mass": invariant_mass(decoded_particles[muon_index], decoded_particles[antimuon_index]),
        "muon": decoded_particles[muon_index],
        "antimuon": decoded_particles[antimuon_index],
        "muon_index": muon_index,
        "antimuon_index": antimuon_index,
        "muon_dr": float(matched_dr[0]),
        "antimuon_dr": float(matched_dr[1]),
        "muon_pid": int(decoded_pid[muon_index]),
        "antimuon_pid": int(decoded_pid[antimuon_index]),
    }
    return original, decoded


def gaussian(x, norm, mean, sigma):
    return norm * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_mass(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[(values >= 60.0) & (values <= 120.0)]
    counts, edges = np.histogram(values, bins=np.arange(60.0, 121.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(values) < 50:
        return {"success": False, "events": int(len(values)), "mean": None, "sigma": None}
    try:
        parameters, _ = curve_fit(
            gaussian,
            centers,
            counts,
            p0=[max(counts.max(), 1), np.median(values), max(np.std(values), 1.0)],
            bounds=([0.0, 80.0, 0.1], [np.inf, 102.0, 20.0]),
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


def summarize(rows, replicas):
    original = np.asarray([row["original_mass"] for row in rows], dtype=float)
    decoded = np.asarray([row["decoded_mass"] for row in rows], dtype=float)
    common = np.isfinite(original) & np.isfinite(decoded)
    original_common, decoded_common = original[common], decoded[common]
    original_fit, decoded_fit = fit_mass(original_common), fit_mass(decoded_common)
    summary = {
        "events": len(rows),
        "original_efficiency": float(np.mean(np.isfinite(original))),
        "decoded_efficiency": float(np.mean(np.isfinite(decoded))),
        "original_matching_efficiency": float(np.mean(np.isfinite(original))),
        "decoded_matching_efficiency": float(np.mean(np.isfinite(decoded))),
        "common_events": int(common.sum()),
        "common_matched_events": int(common.sum()),
        "original_distribution": distribution_moments(original_common),
        "decoded_distribution": distribution_moments(decoded_common),
        "original_fit": original_fit | bootstrap_fit(original_common, replicas),
        "decoded_fit": decoded_fit | bootstrap_fit(decoded_common, replicas),
        "paired_mass_difference_mean": None,
        "paired_mass_difference_std": None,
        "original_central68_half_width": None,
        "decoded_central68_half_width": None,
        "fit_peak_shift": None,
        "fit_width_ratio": None,
    }
    if len(original_common):
        summary["paired_mass_difference_mean"] = float(np.mean(decoded_common - original_common))
        summary["paired_mass_difference_std"] = float(np.std(decoded_common - original_common))
        summary["original_central68_half_width"] = float(
            np.diff(np.quantile(original_common, [0.16, 0.84]))[0] / 2
        )
        summary["decoded_central68_half_width"] = float(
            np.diff(np.quantile(decoded_common, [0.16, 0.84]))[0] / 2
        )
    if original_fit["success"] and decoded_fit["success"]:
        summary["fit_peak_shift"] = decoded_fit["mean"] - original_fit["mean"]
        summary["fit_width_ratio"] = decoded_fit["sigma"] / original_fit["sigma"]
    return summary


def plot_masses(rows, output_dir: Path, candidate_mode="truth"):
    set_mpl_style()
    original = np.asarray([row["original_mass"] for row in rows], dtype=float)
    decoded = np.asarray([row["decoded_mass"] for row in rows], dtype=float)
    common = np.isfinite(original) & np.isfinite(decoded)
    figure, axis = plt.subplots(figsize=(7, 5))
    bins = np.arange(60.0, 121.0, 1.0)
    axis.hist(
        original[common],
        bins=bins,
        histtype="step",
        density=True,
        label=distribution_label("Original", original[common]),
    )
    axis.hist(
        decoded[common],
        bins=bins,
        histtype="step",
        density=True,
        label=distribution_label("Decoded", decoded[common]),
    )
    axis.axvline(Z_MASS_GEV, color="black", linestyle="--", linewidth=1, label=r"$m_Z$")
    axis.set(
        xlabel=r"$Z\to\mu^+\mu^-$ candidate mass [GeV]",
        ylabel="Normalized events",
        title=candidate_mode.replace("_", " "),
    )
    axis.legend(prop={"size": 8})
    figure.tight_layout()
    figure.savefig(output_dir / "z_mumu_mass.png", dpi=180)
    plt.close(figure)
    np.savez_compressed(
        output_dir / "z_mumu_mass_histograms.npz",
        bins=bins,
        original=original[common],
        decoded=decoded[common],
    )


def plot_multirun_masses(results, output_dir: Path, candidate_mode="truth"):
    """Overlay decoded dimuon spectra and a single original reference."""
    set_mpl_style()
    figure, axis = plt.subplots(figsize=(7, 5))
    bins = np.arange(60.0, 121.0, 1.0)
    first_label, first_rows, _ = results[0]
    original = np.asarray([row["original_mass"] for row in first_rows], dtype=float)
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
        decoded = np.asarray([row["decoded_mass"] for row in rows], dtype=float)
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
    axis.axvline(Z_MASS_GEV, color="black", linestyle=":", linewidth=1, label=r"$m_Z$")
    axis.set(
        xlabel=r"$Z\to\mu^+\mu^-$ candidate mass [GeV]",
        ylabel="Normalized events",
        title=f"{candidate_mode.replace('_', ' ')} — decoded comparison",
    )
    axis.legend(prop={"size": 8})
    figure.tight_layout()
    figure.savefig(output_dir / "z_mumu_mass_multirun.png", dpi=180)
    plt.close(figure)
    np.savez_compressed(output_dir / "z_mumu_mass_multirun_histograms.npz", **plot_data)


def upload_to_wandb(args, output_dir: Path, multirun: bool, results):
    """Sync rendered plots and compact replotting inputs as one W&B artifact."""
    if args.no_wandb:
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B synchronization requested, but wandb is not installed.") from exc

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"z-mumu-mass-{output_dir.name}",
        group=args.wandb_group or "orbit-z-mumu-mass",
        entity=args.wandb_entity,
        job_type="multirun-comparison" if multirun else "evaluation",
        config={
            "events": args.events,
            "bootstrap_replicas": args.bootstrap_replicas,
            "candidate_mode": args.candidate_mode,
            "max_abs_eta": args.max_abs_eta,
            "max_match_dr": args.max_match_dr,
        },
    )
    try:
        images = {
            f"z_mumu_mass/{path.stem}": wandb.Image(str(path))
            for path in sorted(output_dir.rglob("*.png"))
        }
        if images:
            run.log(images)
        scalar_metrics = {}
        for label, _, summary in results:
            run_label = output_component(label)
            for representation in ("original", "decoded"):
                moments = summary["z_mumu"][f"{representation}_distribution"]
                for statistic in ("mean", "std"):
                    value = moments[statistic]
                    if value is not None:
                        scalar_metrics[
                            f"mass_moments/{run_label}/{representation}_{statistic}"
                        ] = value
        if scalar_metrics:
            run.log(scalar_metrics)
        artifact = wandb.Artifact(
            name=f"z-mumu-mass-{output_component(output_dir.name)}-{run.id}",
            type="orbit-downstream-evaluation",
            metadata={"multirun": multirun, "events_per_run": args.events},
        )
        for path in sorted(output_dir.rglob("*.npz")) + sorted(output_dir.rglob("*.json")):
            artifact.add_file(str(path), name=str(path.relative_to(output_dir)))
        run.log_artifact(artifact)
    finally:
        wandb.finish()


def candidate_row(prefix: str, candidate):
    if candidate is None:
        return {
            f"{prefix}_mass": float("nan"),
            f"{prefix}_muon_pt": float("nan"),
            f"{prefix}_muon_eta": float("nan"),
            f"{prefix}_muon_phi": float("nan"),
            f"{prefix}_muon_pid": float("nan"),
            f"{prefix}_muon_index": float("nan"),
            f"{prefix}_muon_dr": float("nan"),
            f"{prefix}_antimuon_pt": float("nan"),
            f"{prefix}_antimuon_eta": float("nan"),
            f"{prefix}_antimuon_phi": float("nan"),
            f"{prefix}_antimuon_pid": float("nan"),
            f"{prefix}_antimuon_index": float("nan"),
            f"{prefix}_antimuon_dr": float("nan"),
        }
    muon, antimuon = candidate["muon"], candidate["antimuon"]
    return {
        f"{prefix}_mass": candidate["mass"],
        f"{prefix}_muon_pt": float(muon[2]),
        f"{prefix}_muon_eta": float(muon[0]),
        f"{prefix}_muon_phi": float(muon[1]),
        f"{prefix}_muon_pid": candidate["muon_pid"],
        f"{prefix}_muon_index": candidate["muon_index"],
        f"{prefix}_muon_dr": candidate["muon_dr"],
        f"{prefix}_antimuon_pt": float(antimuon[2]),
        f"{prefix}_antimuon_eta": float(antimuon[0]),
        f"{prefix}_antimuon_phi": float(antimuon[1]),
        f"{prefix}_antimuon_pid": candidate["antimuon_pid"],
        f"{prefix}_antimuon_index": candidate["antimuon_index"],
        f"{prefix}_antimuon_dr": candidate["antimuon_dr"],
    }


def evaluate_run(args, run_dir: Path, output_dir: Path):
    device = torch.device(args.device)
    model, cfg, checkpoint = load_model(
        run_dir,
        device,
        explicit_config=args.config,
        explicit_checkpoint=args.checkpoint,
    )
    sequence_type, max_sequence_length, mask_column, mask_min_value = checkpoint_data_settings(cfg)
    include_energy, energy_shift = checkpoint_energy_settings(cfg)
    batch_size = args.batch_size or (16 if max_sequence_length > 128 else 256)
    prepare_test_time_baseline(model, cfg, output_dir)
    dataset = OrbitParquetDataset(
        args.z_test_manifest,
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
    )

    truth_reader = TruthReader() if args.candidate_mode == "truth" else None
    rows = []
    muon_features = {
        "all_original": [],
        "all_decoded": [],
        "all_pid": [],
        "z_candidates_original": [],
        "z_candidates_decoded": [],
        "z_candidates_pid": [],
    }
    with tqdm(total=args.events, desc="Evaluating Z→μ⁺μ⁻", unit="event") as progress:
        for batch in dataset:
            decoded, decoded_pid = decode_batch(model, batch, device)
            mask = batch["part_mask"].numpy().astype(bool)
            original_pid = batch["part_pid"].numpy()
            for index in range(mask.shape[0]):
                source_file = str(batch["source_files"][index])
                source_row = int(batch["source_rows"][index])
                raw = np.asarray(ak.to_numpy(batch["raw_part_features"][index]), dtype=float)
                valid = mask[index]
                # Raw features remain ragged, whereas ``part_mask`` is padded to the
                # model sequence length. Both retain the leading selected particles.
                particle_count = min(len(raw), int(valid.sum()))
                original = raw[:particle_count, [1, 2, 0]]
                original_pid_event = original_pid[index, valid][:particle_count]
                decoded_event = decoded[index, valid][:particle_count]
                decoded_pid_event = decoded_pid[index, valid][:particle_count]
                original_muons = np.isin(original_pid_event, (MUON_PID, ANTIMUON_PID))
                if np.any(original_muons):
                    muon_features["all_original"].append(original[original_muons])
                    muon_features["all_decoded"].append(decoded_event[original_muons])
                    muon_features["all_pid"].append(original_pid_event[original_muons])
                row = {"source_file": source_file, "source_row": source_row}
                if args.candidate_mode == "truth":
                    truth = decaying_z_to_mumu(truth_reader.row(source_file, source_row))
                    if truth is None:
                        continue
                    original_candidate = truth_matched_dimuon_candidate(
                        original,
                        original_pid_event,
                        truth["muon"],
                        truth["antimuon"],
                        match_dr=args.max_match_dr,
                        max_abs_eta=args.max_abs_eta,
                    )
                    decoded_candidate = truth_matched_dimuon_candidate(
                        decoded_event,
                        decoded_pid_event,
                        truth["muon"],
                        truth["antimuon"],
                        match_dr=args.max_match_dr,
                        max_abs_eta=args.max_abs_eta,
                    )
                    row.update(
                        {
                            "truth_z_pt": float(truth["z"][2]),
                            "truth_z_mass": float(truth["z"][3]),
                        }
                    )
                elif args.candidate_mode == "leading_pt":
                    original_candidate = leading_dimuon_candidate(
                        original, original_pid_event, args.max_abs_eta
                    )
                    decoded_candidate = leading_dimuon_candidate(
                        decoded_event, decoded_pid_event, args.max_abs_eta
                    )
                else:
                    original_candidate, decoded_candidate = cross_matched_dimuon_candidates(
                        original,
                        original_pid_event,
                        decoded_event,
                        decoded_pid_event,
                        max_abs_eta=args.max_abs_eta,
                        max_match_dr=args.max_match_dr,
                    )
                row.update(candidate_row("original", original_candidate))
                row.update(candidate_row("decoded", decoded_candidate))
                if original_candidate is not None and decoded_candidate is not None:
                    muon_features["z_candidates_original"].append(
                        np.stack(
                            (original_candidate["muon"], original_candidate["antimuon"])
                        )
                    )
                    muon_features["z_candidates_decoded"].append(
                        np.stack(
                            (decoded_candidate["muon"], decoded_candidate["antimuon"])
                        )
                    )
                    muon_features["z_candidates_pid"].append(
                        np.asarray((MUON_PID, ANTIMUON_PID), dtype=np.int8)
                    )
                rows.append(row)
                progress.update(1)
                if len(rows) == args.events:
                    break
            if len(rows) == args.events:
                break
    if len(rows) != args.events:
        qualifier = "truth-selected " if args.candidate_mode == "truth" else ""
        raise RuntimeError(f"Expected {args.events} {qualifier}Z→μ⁺μ⁻ events, got {len(rows)}")

    with (output_dir / "z_mumu_candidates.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    compact_muons = {}
    for name, chunks in muon_features.items():
        width = 1 if name.endswith("_pid") else 3
        compact_muons[name] = (
            np.concatenate(chunks).astype(np.int8 if width == 1 else np.float32)
            if chunks
            else np.empty((0,) if width == 1 else (0, width), dtype=np.int8 if width == 1 else np.float32)
        )
    np.savez_compressed(output_dir / "z_mumu_muon_features.npz", **compact_muons)
    summary = {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "events": args.events,
        "sequence_type": sequence_type,
        "max_sequence_length": max_sequence_length,
        "candidate_mode": args.candidate_mode,
        "particle_selection": {"mask_column": mask_column, "mask_min_value": mask_min_value},
        "candidate_selection": {
            "original_pid_classes": [MUON_PID, ANTIMUON_PID],
            "decoded_pid_classes": [MUON_PID, ANTIMUON_PID],
            "pair": (
                "independent leading-pT muon and antimuon"
                if args.candidate_mode == "leading_pt"
                else "original leading muon pair matched to decoded muons by Hungarian delta-R"
                if args.candidate_mode == "hungarian"
                else "legacy independent Hungarian matching to direct generator daughters"
            ),
            "max_abs_eta": args.max_abs_eta,
            "max_match_dr": args.max_match_dr,
            "pid_constraint": "muon and antimuon candidates retain their corresponding PID class",
            "muon_mass_gev": MUON_MASS_GEV,
        },
        "z_mumu": summarize(rows, args.bootstrap_replicas),
    }
    (output_dir / "z_mumu_mass_metrics.json").write_text(json.dumps(summary, indent=2))
    plot_masses(rows, output_dir, args.candidate_mode)
    return rows, summary


def main():
    args = parse_args()
    if (args.config is None) != (args.checkpoint is None):
        raise ValueError("--config and --checkpoint must be provided together")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
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
        print(f"Wrote Z→μ⁺μ⁻ mass evaluation for {label!r} to {run_output_dir}")
    if multirun:
        plot_multirun_masses(results, args.output_dir, args.candidate_mode)
        comparison = {
            "z_test_manifest": str(args.z_test_manifest.resolve()),
            "events_per_run": args.events,
            "runs": [
                {"label": label, "run_dir": str(run_dir), "metrics": summary}
                for (run_dir, label), (_, _, summary) in zip(runs, results, strict=True)
            ],
        }
        (args.output_dir / "z_mumu_mass_multirun_metrics.json").write_text(
            json.dumps(comparison, indent=2)
        )
        print(f"Wrote Z→μ⁺μ⁻ multi-run comparison to {args.output_dir}")
    upload_to_wandb(args, args.output_dir, multirun, results)


if __name__ == "__main__":
    main()
