#!/usr/bin/env python3
"""Cache compact PID confusion counts for selected 5M campaign models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT / "vqtorch", PROJECT_ROOT):
    sys.path.insert(0, str(root))

import torch
import yaml
from tqdm.auto import tqdm

from gabbro.data.orbit_parquet import PID_CLASS_NAMES
from scripts.evaluate_orbit_5m_plot_cache import decode, iter_batches, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", action="append", required=True)
    parser.add_argument(
        "--suite",
        action="append",
        choices=("ttbar", "sm_mixture", "gghbb", "minbias"),
        required=True,
    )
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(model, cfg, manifest_dir, suite, events, batch_size, device):
    counts = torch.zeros((len(PID_CLASS_NAMES), len(PID_CLASS_NAMES)), dtype=torch.int64)
    processed_events = 0
    progress = tqdm(total=events, desc=f"PID confusion: {suite}", unit="event")
    for batch in iter_batches(cfg, manifest_dir, suite, events, batch_size):
        features = batch["part_features"].to(device)
        mask = batch["part_mask"].to(device)
        pid = batch["part_pid"].to(device)
        with torch.inference_mode():
            _, _, _, pid_logits = decode(model, features, mask, pid)
        if pid_logits is None:
            raise ValueError("PID-enabled model did not return PID logits")
        truth = pid[mask].to(torch.int64)
        prediction = pid_logits.argmax(dim=-1)[mask].to(torch.int64)
        encoded = truth * len(PID_CLASS_NAMES) + prediction
        counts += torch.bincount(
            encoded.detach().cpu(), minlength=len(PID_CLASS_NAMES) ** 2
        ).reshape(len(PID_CLASS_NAMES), len(PID_CLASS_NAMES))
        processed_events += int(mask.shape[0])
        progress.update(int(mask.shape[0]))
    progress.close()
    if processed_events != events:
        raise RuntimeError(f"Expected {events} events, processed {processed_events}")
    return counts.numpy(), processed_events


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    registry = yaml.safe_load(registry_path.read_text())["models"]
    by_id = {row["id"]: row for row in registry}
    missing = sorted(set(args.model_id) - set(by_id))
    if missing:
        raise ValueError(f"Unknown model IDs: {missing}")
    if args.events < 1 or args.batch_size < 1:
        raise ValueError("--events and --batch-size must be positive")
    device = torch.device(args.device)
    for model_id in args.model_id:
        spec = by_id[model_id]
        model, cfg = load_model(spec, device)
        for suite in args.suite:
            counts, processed_events = evaluate(
                model,
                cfg,
                args.manifest_dir.resolve(),
                suite,
                args.events,
                args.batch_size,
                device,
            )
            row_totals = counts.sum(axis=1, keepdims=True)
            normalized = counts / row_totals.clip(min=1)
            target = (
                args.output_dir.resolve()
                / model_id
                / suite
                / "artifacts"
                / "pid_confusion_matrix_test_suite_0.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "model_id": model_id,
                        "model": spec,
                        "suite": suite,
                        "events": processed_events,
                        "particles": int(counts.sum()),
                        "truth_labels": list(PID_CLASS_NAMES),
                        "predicted_labels": list(PID_CLASS_NAMES),
                        "counts": counts.tolist(),
                        "row_normalized": normalized.tolist(),
                        "registry": {
                            "path": str(registry_path),
                            "sha256": sha256(registry_path),
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Saved {target}")


if __name__ == "__main__":
    main()
