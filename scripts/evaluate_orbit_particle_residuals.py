#!/usr/bin/env python
"""Cache inclusive physical particle residuals for an ORBIT checkpoint."""

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

import numpy as np
import torch
from tqdm.auto import tqdm

from gabbro.data.orbit_parquet import OrbitParquetDataset
from gabbro.plotting.orbit import collect_physical_reconstruction_histograms
from scripts.evaluate_orbit_pid_pulls import (
    checkpoint_data_settings,
    load_model,
    parse_test_classes,
    split_event_quota,
    to_physical,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--test-class",
        action="append",
        required=True,
        metavar="CLASS=MANIFEST[,MANIFEST...]",
        help="Repeat for balanced top-level classes; manifests within a class are balanced.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.events < 1:
        raise ValueError("--events must be positive")
    classes = parse_test_classes(args.test_class)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, checkpoint = load_model(args.run_dir.resolve(), device)
    sequence_type, max_length, mask_column, mask_min_value = checkpoint_data_settings(cfg)
    batch_size = args.batch_size or (16 if max_length > 128 else 256)

    original_chunks: list[np.ndarray] = []
    decoded_chunks: list[np.ndarray] = []
    processed_events = 0
    processed_particles = 0
    started = time.perf_counter()
    with tqdm(total=args.events, desc="Caching particle residuals", unit="event") as progress:
        for (_, manifests), class_quota in zip(
            classes.items(), split_event_quota(args.events, len(classes)), strict=True
        ):
            for manifest, manifest_quota in zip(
                manifests, split_event_quota(class_quota, len(manifests)), strict=True
            ):
                dataset = OrbitParquetDataset(
                    manifest,
                    sequence_type=sequence_type,
                    max_sequence_length=max_length,
                    batch_size=batch_size,
                    shuffle_row_groups=False,
                    mask_column=mask_column,
                    mask_min_value=mask_min_value,
                    max_events=manifest_quota,
                    pid_cfg=cfg.get("pid"),
                    include_energy=bool(cfg.data.get("include_energy", False)),
                    energy_shift=float(cfg.data.get("energy_shift", 2.5)),
                )
                for batch in dataset:
                    features = batch["part_features"].to(device)
                    mask_tensor = batch["part_mask"].to(device)
                    pid = batch.get("part_pid")
                    if pid is not None:
                        pid = pid.to(device)
                    with torch.inference_mode():
                        reconstructed, _ = model(features, mask_tensor, pid_particle=pid)
                    mask = mask_tensor.cpu().numpy().astype(bool)
                    original = to_physical(features.cpu().numpy())[mask]
                    decoded = to_physical(reconstructed.cpu().numpy())[mask]
                    original_chunks.append(original.astype(np.float32, copy=False))
                    decoded_chunks.append(decoded.astype(np.float32, copy=False))
                    processed_particles += len(original)
                    batch_events = len(mask)
                    processed_events += batch_events
                    progress.update(batch_events)

    original = np.concatenate(original_chunks)
    decoded = np.concatenate(decoded_chunks)
    histograms = collect_physical_reconstruction_histograms(
        ["Eta", "Phi", "pT"], original, decoded
    )
    output = args.output_dir / "ttbar_particle_residual_histograms.npz"
    np.savez_compressed(
        output,
        **{
            f"physical_{key}": value
            for key, value in histograms.items()
            if key.endswith("_diff_counts") or key.endswith("_diff_bins")
        },
    )
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint),
        "test_classes": {
            name: [str(path) for path in manifests]
            for name, manifests in classes.items()
        },
        "processed_events": processed_events,
        "processed_particles": processed_particles,
        "batch_size": batch_size,
        "residual_definition": "decoded - original; phi is wrapped to [-pi, pi)",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "ttbar_particle_residual_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
