#!/usr/bin/env python
"""Compare physical particle kinematics decoded by several ORBIT checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "vqtorch", PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from gabbro.data.orbit_transforms import KinematicTransform
from gabbro.plotting.orbit import (
    TRUTH_REFERENCE_COLOR,
    TRUTH_REFERENCE_FILL_ALPHA,
    multirun_color,
    multirun_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        required=True,
        metavar=("RUN_DIR", "LABEL"),
        help="Checkpoint run and its display label; repeat for each model.",
    )
    parser.add_argument(
        "--evaluation-run-dir",
        type=Path,
        required=True,
        help="Run whose resolved data config defines the evaluation sample.",
    )
    parser.add_argument("--suite", default="training_like")
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolved_config_path(run_dir: Path) -> Path:
    for candidate in (run_dir / "config_resolved.yaml", run_dir / ".hydra/config.yaml"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No resolved or Hydra config found below {run_dir}")


def load_model(run_dir: Path, device: torch.device):
    cfg = OmegaConf.load(resolved_config_path(run_dir))
    checkpoint = run_dir / "checkpoints/best.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
    model = instantiate(cfg.model)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def evaluation_loader(run_dir: Path, suite: str, events: int, num_workers: int):
    cfg = OmegaConf.load(resolved_config_path(run_dir))
    if suite not in cfg.data.test_suites:
        raise ValueError(f"Unknown test suite {suite!r}; choose from {list(cfg.data.test_suites)}")
    suite_cfg = OmegaConf.to_container(cfg.data.test_suites[suite], resolve=True)
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    data_cfg.test_suites = {suite: suite_cfg}
    data_cfg.test_suites[suite].event_budget = events
    data_cfg.num_workers = num_workers
    datamodule = instantiate(data_cfg)
    loaders = datamodule.test_dataloader()
    if len(loaders) != 1:
        raise RuntimeError(f"Expected one selected test suite, got {len(loaders)}")
    return loaders[0]


def run_kinematics(run_dir: Path) -> KinematicTransform:
    cfg = OmegaConf.load(resolved_config_path(run_dir))
    return KinematicTransform.from_data_config(cfg.data)


def to_physical(values: np.ndarray, transform: KinematicTransform) -> dict[str, np.ndarray]:
    physical = transform.to_physical(values)
    return {"eta": physical[..., 0], "phi": physical[..., 1], "pt": physical[..., 2]}


def histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=edges, density=True)
    return np.nan_to_num(counts)


def plot_panel(axis, key: str, payload: dict, bins: int) -> dict[str, np.ndarray]:
    specifications = {
        "pt": (np.geomspace(0.05, 2_000.0, bins + 1), r"Particle $p_T$ [GeV]"),
        "eta": (np.linspace(-4.5, 4.5, bins + 1), r"Particle $\eta$"),
        "phi": (np.linspace(-np.pi, np.pi, bins + 1), r"Particle $\phi$"),
    }
    edges, xlabel = specifications[key]
    original_counts = histogram(payload["original"][key], edges)
    axis.stairs(
        original_counts,
        edges,
        fill=True,
        color=TRUTH_REFERENCE_COLOR,
        alpha=TRUTH_REFERENCE_FILL_ALPHA,
        linewidth=1.2,
    )
    saved = {f"{key}_bins": edges, f"{key}_original_counts": original_counts}
    centers = np.sqrt(edges[:-1] * edges[1:]) if key == "pt" else (edges[:-1] + edges[1:]) / 2
    marker_indices = np.arange(2, len(centers), 7)
    for decoded in payload["decoded"]:
        counts = histogram(decoded["values"][key], edges)
        color = multirun_color(decoded["style"])
        marker = multirun_marker(decoded["style"])
        axis.stairs(counts, edges, color=color, linewidth=2.2)
        axis.plot(
            centers[marker_indices],
            counts[marker_indices],
            linestyle="none",
            marker=marker,
            markersize=7,
            color=color,
        )
        slug = decoded["style"].lower().replace(" ", "_")
        saved[f"{key}_{slug}_counts"] = counts
    if key == "pt":
        axis.set_xscale("log")
        axis.set_yscale("log", nonpositive="clip")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Normalized density")
    axis.set_xlim(edges[0], edges[-1])
    return saved


def legend_handles(payload: dict) -> list:
    handles = [
        Patch(
            facecolor=TRUTH_REFERENCE_COLOR,
            alpha=TRUTH_REFERENCE_FILL_ALPHA,
            label="Original SM mixture",
        )
    ]
    for decoded in payload["decoded"]:
        handles.append(
            Line2D(
                [],
                [],
                color=multirun_color(decoded["style"]),
                marker=multirun_marker(decoded["style"]),
                linewidth=2.2,
                markersize=10,
                label=decoded["label"],
            )
        )
    return handles


def save_plots(payload: dict, output_dir: Path, bins: int) -> None:
    hep.style.use("CMS")
    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_payload = {}
    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)
    for axis, key in zip(axes, ("pt", "eta", "phi")):
        histogram_payload.update(plot_panel(axis, key, payload, bins))
    axes[0].legend(handles=legend_handles(payload), fontsize=17, frameon=False)
    figure.savefig(
        output_dir / "physical_particle_pt_eta_phi.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)

    for key in ("pt", "eta", "phi"):
        figure, axis = plt.subplots(figsize=(7.6, 6.2))
        plot_panel(axis, key, payload, bins)
        axis.legend(handles=legend_handles(payload), fontsize=16, frameon=False)
        figure.savefig(
            output_dir / f"physical_particle_{key}.png", dpi=220, bbox_inches="tight"
        )
        plt.close(figure)
    np.savez_compressed(output_dir / "physical_particle_pt_eta_phi_histograms.npz", **histogram_payload)


def main() -> None:
    args = parse_args()
    if args.events < 1 or args.bins < 1:
        raise ValueError("--events and --bins must be positive")
    device = torch.device(args.device)
    run_specs = [(Path(path).resolve(), label) for path, label in args.run]
    models = []
    checkpoints = []
    for run_dir, label in run_specs:
        model, checkpoint = load_model(run_dir, device)
        models.append((label, model))
        checkpoints.append(str(checkpoint))

    # Every model is fed the evaluation run's preprocessed features, so all of
    # them must have been trained with the same kinematic transform.
    kinematics = run_kinematics(args.evaluation_run_dir.resolve())
    for run_dir, label in run_specs:
        if run_kinematics(run_dir) != kinematics:
            raise ValueError(
                f"Run {label!r} uses {run_kinematics(run_dir)}, but the evaluation run uses "
                f"{kinematics}; this script needs a shared preprocessing."
            )

    original_chunks = []
    decoded_chunks = {label: [] for label, _ in models}
    processed_events = 0
    loader = evaluation_loader(
        args.evaluation_run_dir.resolve(), args.suite, args.events, args.num_workers
    )
    with tqdm(total=args.events, desc="Decoding physical particle kinematics", unit="event") as bar:
        for batch in loader:
            features = batch["part_features"].to(device)
            mask = batch["part_mask"].to(device)
            pid = batch.get("part_pid")
            if pid is not None:
                pid = pid.to(device)
            mask_numpy = mask.detach().cpu().numpy().astype(bool)
            original_chunks.append(features.detach().cpu().numpy()[mask_numpy])
            for label, model in models:
                with torch.inference_mode():
                    reconstructed, _ = model.forward(features, mask, pid_particle=pid)
                decoded_chunks[label].append(reconstructed.detach().cpu().numpy()[mask_numpy])
            batch_events = len(mask_numpy)
            processed_events += batch_events
            bar.update(batch_events)

    payload = {
        "original": to_physical(np.concatenate(original_chunks), kinematics),
        "decoded": [
            {
                "label": label,
                "style": label,
                "values": to_physical(np.concatenate(decoded_chunks[label]), kinematics),
            }
            for label, _ in models
        ],
    }
    save_plots(payload, args.output_dir, args.bins)
    summary = {
        "evaluation_run_dir": str(args.evaluation_run_dir.resolve()),
        "test_suite": args.suite,
        "events": processed_events,
        "particles": int(len(payload["original"]["pt"])),
        "runs": [
            {"run_dir": str(run_dir), "label": label, "checkpoint": checkpoint}
            for (run_dir, label), checkpoint in zip(run_specs, checkpoints)
        ],
        "styles": {
            label: {
                "color": multirun_color(label),
                "marker": multirun_marker(label),
            }
            for _, label in run_specs
        },
    }
    (args.output_dir / "physical_particle_pt_eta_phi_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
