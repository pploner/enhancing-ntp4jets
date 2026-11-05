#!/usr/bin/env python
"""Compare original and decoded tau32 for boosted all-hadronic tt AK8 jets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import awkward as ak
import fastjet
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import vector
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from gabbro.data.orbit_transforms import KinematicTransform
from gabbro.callbacks.orbit_plotting_callback import (
    OrbitPlottingCallback,
    match_jets_by_delta_r,
)
from gabbro.data.orbit_parquet import OrbitParquetDataset, SEQUENCE_SCHEMAS

vector.register_awkward()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--jet-min-pt", type=float, default=450.0)
    parser.add_argument("--decoded-jet-min-pt", type=float, default=30.0)
    parser.add_argument("--jet-count-min-pt", type=float, default=450.0)
    parser.add_argument("--jet-radius", type=float, default=0.8)
    parser.add_argument("--max-match-dr", type=float, default=0.4)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--reuse-tau32-cache",
        type=Path,
        help=(
            "Reuse original_tau32, decoded_tau32, and match_delta_r from an existing "
            "NPZ while recomputing only the per-event AK8 jet counts."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(run_dir: Path, device: torch.device):
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        config_path = run_dir / ".hydra/config.yaml"
    checkpoint = run_dir / "checkpoints/best.ckpt"
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Expected a config and checkpoints/best.ckpt below {run_dir}")
    cfg = OmegaConf.load(config_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
    model = instantiate(cfg.model)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), cfg, checkpoint


def checkpoint_data_settings(cfg):
    sequence_type = str(cfg.data.get("sequence_type", "particle"))
    if not sequence_type.startswith("particle"):
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


def to_physical(values: np.ndarray, transform: KinematicTransform) -> np.ndarray:
    """Undo the run's kinematic preprocessing: ``[..., >=4]`` -> ``[eta, phi, pT]``."""
    return transform.to_physical(values)


def reconstruct_jets(values: np.ndarray, radius: float, min_pt: float):
    if not len(values):
        return np.empty(0), np.empty(0), np.empty(0), []
    particles = ak.zip(
        {
            "pt": [values[:, 2]],
            "eta": [values[:, 0]],
            "phi": [values[:, 1]],
            "mass": [np.zeros(len(values))],
        },
        with_name="Momentum4D",
    )
    cluster = fastjet.ClusterSequence(
        particles,
        fastjet.JetDefinition(fastjet.antikt_algorithm, radius),
    )
    jets = cluster.inclusive_jets(min_pt=min_pt)
    constituents = cluster.constituents(min_pt=min_pt)[0]
    return (
        np.asarray(jets.pt[0]),
        np.asarray(jets.eta[0]),
        np.asarray(jets.phi[0]),
        constituents,
    )


def tau32(constituents, radius: float) -> np.ndarray:
    return np.asarray(
        [
            OrbitPlottingCallback._calculate_tau32(parts[np.newaxis], radius)
            for parts in constituents
        ],
        dtype=np.float64,
    )


def plot_histogram(
    original: np.ndarray,
    decoded: np.ndarray,
    original_jet_counts: np.ndarray,
    decoded_jet_counts: np.ndarray,
    output: Path,
    bins: int,
    jet_count_min_pt: float,
):
    hep.style.use("CMS")
    figure, (count_axis, axis) = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.2),
        gridspec_kw={"width_ratios": (1.0, 1.2)},
    )
    edges = np.linspace(0.0, 1.2, bins + 1)
    axis.hist(
        original,
        bins=edges,
        density=True,
        histtype="stepfilled",
        alpha=0.38,
        color="#3f90da",
        linewidth=1.5,
        label="Original",
    )
    axis.hist(
        decoded,
        bins=edges,
        density=True,
        histtype="step",
        color="#ffa90e",
        linewidth=2.2,
        label="Decoded",
    )
    axis.set_xlabel(r"AK8 jet $\tau_{32}$")
    axis.set_ylabel("Normalized density")
    axis.set_xlim(edges[0], edges[-1])
    axis.legend(fontsize=19, frameon=False)

    maximum_count = max(
        int(np.max(original_jet_counts, initial=0)),
        int(np.max(decoded_jet_counts, initial=0)),
    )
    count_edges = np.arange(-0.5, maximum_count + 1.5, 1.0)
    count_axis.hist(
        original_jet_counts,
        bins=count_edges,
        density=True,
        histtype="stepfilled",
        alpha=0.38,
        color="#3f90da",
        linewidth=1.5,
        label="Original",
    )
    count_axis.hist(
        decoded_jet_counts,
        bins=count_edges,
        density=True,
        histtype="step",
        color="#ffa90e",
        linewidth=2.2,
        label="Decoded",
    )
    count_axis.set_xlabel(
        rf"Number of AK8 jets ($p_{{\mathrm{{T}}}} \geq {jet_count_min_pt:g}$ GeV)"
    )
    count_axis.set_ylabel("Normalized events")
    count_axis.set_yscale("log")
    count_axis.set_xlim(count_edges[0], count_edges[-1])
    count_axis.legend(fontsize=19, frameon=False)

    figure.subplots_adjust(wspace=0.30)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.events < 1 or args.bins < 1:
        raise ValueError("--events and --bins must be positive")
    if (
        args.jet_min_pt <= 0
        or args.decoded_jet_min_pt < 0
        or args.jet_count_min_pt < 0
    ):
        raise ValueError("Jet pT thresholds must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, checkpoint = load_model(args.run_dir.resolve(), device)
    sequence_type, max_length, mask_column, mask_min_value = checkpoint_data_settings(cfg)
    batch_size = args.batch_size or (16 if max_length > 128 else 256)
    dataset = OrbitParquetDataset(
        args.test_manifest.resolve(),
        sequence_type=sequence_type,
        max_sequence_length=max_length,
        batch_size=batch_size,
        shuffle_row_groups=False,
        mask_column=mask_column,
        mask_min_value=mask_min_value,
        event_filter_sequence_type="jet_ak8",
        event_filter_min_pt=args.jet_min_pt,
        max_events=args.events,
        pid_cfg=cfg.get("pid"),
        include_energy=bool(cfg.data.get("include_energy", False)),
        energy_shift=float(cfg.data.get("energy_shift", 2.5)),
        **KinematicTransform.from_data_config(cfg.data).as_kwargs(),
    )

    if args.reuse_tau32_cache is not None:
        cache_path = args.reuse_tau32_cache.resolve()
        with np.load(cache_path) as cache:
            original_tau = np.asarray(cache["original_tau32"], dtype=np.float64).copy()
            decoded_tau = np.asarray(cache["decoded_tau32"], dtype=np.float64).copy()
            match_delta_r = np.asarray(cache["match_delta_r"], dtype=np.float64).copy()
    else:
        cache_path = None
        original_tau = []
        decoded_tau = []
        match_delta_r = []
    original_jet_counts = []
    decoded_jet_counts = []
    original_jets = 0
    decoded_jets = 0
    processed_events = 0
    started = time.perf_counter()
    kinematics = KinematicTransform.from_data_config(cfg.data)
    with tqdm(total=args.events, desc="Evaluating boosted all-hadronic tt", unit="event") as bar:
        for batch in dataset:
            features = batch["part_features"].to(device)
            mask_tensor = batch["part_mask"].to(device)
            pid = batch.get("part_pid")
            if pid is not None:
                pid = pid.to(device)
            with torch.inference_mode():
                reconstructed, _ = model(features, mask_tensor, pid_particle=pid)
            original = to_physical(features.detach().cpu().numpy(), kinematics)
            decoded = to_physical(reconstructed.detach().cpu().numpy(), kinematics)
            mask = mask_tensor.detach().cpu().numpy().astype(bool)

            for event_index in range(len(mask)):
                orig_pt_all, orig_eta_all, orig_phi_all, orig_const_all = reconstruct_jets(
                    original[event_index, mask[event_index]],
                    args.jet_radius,
                    min(args.jet_min_pt, args.jet_count_min_pt),
                )
                reco_pt_all, reco_eta_all, reco_phi_all, reco_const_all = reconstruct_jets(
                    decoded[event_index, mask[event_index]],
                    args.jet_radius,
                    min(args.decoded_jet_min_pt, args.jet_count_min_pt),
                )
                original_jet_counts.append(int(np.sum(orig_pt_all >= args.jet_count_min_pt)))
                decoded_jet_counts.append(int(np.sum(reco_pt_all >= args.jet_count_min_pt)))
                if cache_path is not None:
                    original_jets += int(np.sum(orig_pt_all >= args.jet_min_pt))
                    decoded_jets += int(np.sum(reco_pt_all >= args.decoded_jet_min_pt))
                    continue
                orig_tau_mask = orig_pt_all >= args.jet_min_pt
                reco_tau_mask = reco_pt_all >= args.decoded_jet_min_pt
                orig_pt = orig_pt_all[orig_tau_mask]
                orig_eta = orig_eta_all[orig_tau_mask]
                orig_phi = orig_phi_all[orig_tau_mask]
                orig_const = [
                    constituents
                    for constituents, keep in zip(orig_const_all, orig_tau_mask)
                    if keep
                ]
                reco_pt = reco_pt_all[reco_tau_mask]
                reco_eta = reco_eta_all[reco_tau_mask]
                reco_phi = reco_phi_all[reco_tau_mask]
                reco_const = [
                    constituents
                    for constituents, keep in zip(reco_const_all, reco_tau_mask)
                    if keep
                ]
                original_jets += len(orig_pt)
                decoded_jets += len(reco_pt)
                orig_idx, reco_idx, delta_r = match_jets_by_delta_r(
                    orig_eta,
                    orig_phi,
                    reco_eta,
                    reco_phi,
                    max_delta_r=args.max_match_dr,
                )
                if len(orig_idx):
                    orig_values = tau32([orig_const[index] for index in orig_idx], args.jet_radius)
                    reco_values = tau32([reco_const[index] for index in reco_idx], args.jet_radius)
                    finite = np.isfinite(orig_values) & np.isfinite(reco_values)
                    original_tau.extend(orig_values[finite])
                    decoded_tau.extend(reco_values[finite])
                    match_delta_r.extend(delta_r[finite])
            processed_events += len(mask)
            bar.update(len(mask))

    original_tau = np.asarray(original_tau)
    decoded_tau = np.asarray(decoded_tau)
    match_delta_r = np.asarray(match_delta_r)
    original_jet_counts = np.asarray(original_jet_counts, dtype=np.int32)
    decoded_jet_counts = np.asarray(decoded_jet_counts, dtype=np.int32)
    if not len(original_tau):
        raise RuntimeError("No matched finite-tau32 AK8 jets passed the requested selection")
    plot_path = args.output_dir / "tt_hadronic_ak8_pt450_tau32.png"
    plot_histogram(
        original_tau,
        decoded_tau,
        original_jet_counts,
        decoded_jet_counts,
        plot_path,
        args.bins,
        args.jet_count_min_pt,
    )
    np.savez_compressed(
        args.output_dir / "tt_hadronic_ak8_pt450_tau32_values.npz",
        original_tau32=original_tau,
        decoded_tau32=decoded_tau,
        match_delta_r=match_delta_r,
        original_ak8_jet_count=original_jet_counts,
        decoded_ak8_jet_count=decoded_jet_counts,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint),
        "test_manifest": str(args.test_manifest.resolve()),
        "selection": {
            "process": "tt0123j_5f_ckm_LO_MLM_hadronic",
            "source_event_filter": f"at least one stored AK8 jet with pT >= {args.jet_min_pt:g} GeV",
            "jet_algorithm": "anti-kt",
            "jet_radius": args.jet_radius,
            "original_jet_min_pt_gev": args.jet_min_pt,
            "decoded_jet_min_pt_gev": args.decoded_jet_min_pt,
            "jet_count_min_pt_gev": args.jet_count_min_pt,
            "matching": "Hungarian delta-R",
            "max_match_delta_r": args.max_match_dr,
            "reused_tau32_cache": str(cache_path) if cache_path is not None else None,
        },
        "processed_events": processed_events,
        "original_selected_jets": original_jets,
        "decoded_candidate_jets": decoded_jets,
        "matched_finite_tau32_jets": int(len(original_tau)),
        "original_mean_ak8_jet_count": float(np.mean(original_jet_counts)),
        "decoded_mean_ak8_jet_count": float(np.mean(decoded_jet_counts)),
        "match_efficiency": float(len(original_tau) / original_jets) if original_jets else 0.0,
        "original_tau32": {
            "mean": float(np.mean(original_tau)),
            "std": float(np.std(original_tau)),
        },
        "decoded_tau32": {
            "mean": float(np.mean(decoded_tau)),
            "std": float(np.std(decoded_tau)),
        },
        "decoded_minus_original_tau32": {
            "mean": float(np.mean(decoded_tau - original_tau)),
            "std": float(np.std(decoded_tau - original_tau)),
        },
        "match_delta_r_mean": float(np.mean(match_delta_r)),
        "elapsed_seconds": elapsed,
        "events_per_second": processed_events / elapsed,
    }
    (args.output_dir / "tt_hadronic_ak8_pt450_tau32_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved {plot_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
