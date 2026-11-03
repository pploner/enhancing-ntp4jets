#!/usr/bin/env python
"""Cache matched anti-kt jet residuals for an ORBIT checkpoint."""

from __future__ import annotations

import argparse
import json
import math
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

from gabbro.callbacks.orbit_plotting_callback import match_jets_by_delta_r
from gabbro.data.orbit_parquet import OrbitParquetDataset
from scripts.evaluate_orbit_tau32 import (
    checkpoint_data_settings,
    load_model,
    reconstruct_jets,
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
    parser.add_argument("--jet-radius", type=float, default=0.4)
    parser.add_argument("--jet-min-pt", type=float, default=30.0)
    parser.add_argument("--max-match-dr", type=float, default=0.2)
    parser.add_argument("--leading-jets", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_classes(values: list[str]) -> dict[str, list[Path]]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CLASS=MANIFEST[,MANIFEST...], got {value!r}")
        name, manifests = value.split("=", 1)
        paths = [Path(item).resolve() for item in manifests.split(",") if item]
        if not name or not paths or any(not path.is_file() for path in paths):
            raise ValueError(f"Invalid class specification: {value!r}")
        result[name] = paths
    return result


def split_quota(total: int, count: int) -> list[int]:
    return [total // count + (index < total % count) for index in range(count)]


def wrapped_delta_phi(decoded: np.ndarray, original: np.ndarray) -> np.ndarray:
    return np.remainder(decoded - original + np.pi, 2 * np.pi) - np.pi


def append_matches(storage, prefix, original, decoded, original_indices, decoded_indices):
    if not len(original_indices):
        return
    original_pt, original_eta, original_phi = original
    decoded_pt, decoded_eta, decoded_phi = decoded
    original_pt = original_pt[original_indices]
    decoded_pt = decoded_pt[decoded_indices]
    storage[f"{prefix}_delta_pt"].extend((decoded_pt - original_pt) / original_pt)
    storage[f"{prefix}_delta_eta"].extend(decoded_eta[decoded_indices] - original_eta[original_indices])
    storage[f"{prefix}_delta_phi"].extend(wrapped_delta_phi(decoded_phi[decoded_indices], original_phi[original_indices]))


def main() -> None:
    args = parse_args()
    if args.events < 1 or args.leading_jets < 1:
        raise ValueError("--events and --leading-jets must be positive")
    classes = parse_classes(args.test_class)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, checkpoint = load_model(args.run_dir.resolve(), device)
    sequence_type, max_length, mask_column, mask_min_value = checkpoint_data_settings(cfg)
    batch_size = args.batch_size or (16 if max_length > 128 else 256)
    class_quotas = split_quota(args.events, len(classes))
    storage = {f"{prefix}_delta_{feature}": [] for prefix in ("all", "leading5") for feature in ("pt", "eta", "phi")}
    match_dr = {"all": [], "leading5": []}
    processed = 0
    started = time.perf_counter()
    with tqdm(total=args.events, desc="Caching matched jet residuals", unit="event") as progress:
        for (class_name, manifests), class_quota in zip(classes.items(), class_quotas, strict=True):
            for manifest, manifest_quota in zip(manifests, split_quota(class_quota, len(manifests)), strict=True):
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
                    original = to_physical(features.cpu().numpy())
                    decoded = to_physical(reconstructed.cpu().numpy())
                    mask = mask_tensor.cpu().numpy().astype(bool)
                    for event_index in range(len(mask)):
                        original_jets = reconstruct_jets(original[event_index, mask[event_index]], args.jet_radius, args.jet_min_pt)[:3]
                        decoded_jets = reconstruct_jets(decoded[event_index, mask[event_index]], args.jet_radius, args.jet_min_pt)[:3]
                        all_orig_idx, all_dec_idx, all_dr = match_jets_by_delta_r(original_jets[1], original_jets[2], decoded_jets[1], decoded_jets[2], max_delta_r=args.max_match_dr)
                        append_matches(storage, "all", original_jets, decoded_jets, all_orig_idx, all_dec_idx)
                        match_dr["all"].extend(all_dr)
                        leading_order = np.argsort(original_jets[0])[::-1][: args.leading_jets]
                        lead_orig_idx, lead_dec_idx, lead_dr = match_jets_by_delta_r(original_jets[1][leading_order], original_jets[2][leading_order], decoded_jets[1], decoded_jets[2], max_delta_r=args.max_match_dr)
                        append_matches(storage, "leading5", original_jets, decoded_jets, leading_order[lead_orig_idx], lead_dec_idx)
                        match_dr["leading5"].extend(lead_dr)
                    processed += len(mask)
                    progress.update(len(mask))
    arrays = {key: np.asarray(values, dtype=np.float32) for key, values in storage.items()}
    arrays.update({f"{prefix}_match_delta_r": np.asarray(values, dtype=np.float32) for prefix, values in match_dr.items()})
    output = args.output_dir / "sm_mixture_jet_residuals.npz"
    np.savez_compressed(output, **arrays)
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint),
        "test_classes": {key: [str(path) for path in value] for key, value in classes.items()},
        "processed_events": processed,
        "jet_algorithm": "anti-kt",
        "jet_radius": args.jet_radius,
        "jet_min_pt_gev": args.jet_min_pt,
        "matching": "Hungarian delta-R",
        "max_match_delta_r": args.max_match_dr,
        "leading_jets": args.leading_jets,
        "all_matches": len(arrays["all_delta_pt"]),
        "leading_matches": len(arrays["leading5_delta_pt"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "sm_mixture_jet_residuals.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
