#!/usr/bin/env python3
"""Measure selection-specific inverse-frequency PID reconstruction weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_EXPERIMENTS = {
    "full500": "orbit_campaign_full500_sm_vq4096",
    "puppi128": "orbit_campaign_puppi128_sm_vq4096",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        choices=("all", *SELECTION_EXPERIMENTS),
        default="all",
    )
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "ORBIT_MANIFEST_DIR", REPO_ROOT / "manifests" / "production_final"
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "campaigns" / "orbit_full_statistics_pid_weights.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compose_data(experiment: str, events: int, num_workers: int):
    if not OmegaConf.has_resolver("nodename_bigram"):
        OmegaConf.register_new_resolver("nodename_bigram", lambda: "pidweights")
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment={experiment}",
                f"data.train_event_budget={events}",
                "data.val_event_budget=1",
                f"data.num_workers={num_workers}",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        OmegaConf.resolve(cfg.data)
    return cfg.data


def estimate(selection: str, events: int, num_workers: int) -> dict:
    data_cfg = compose_data(SELECTION_EXPERIMENTS[selection], events, num_workers)
    datamodule = hydra.utils.instantiate(data_cfg)
    counts = torch.zeros(8, dtype=torch.int64)
    observed_events = 0
    for batch in datamodule.train_dataloader():
        mask = batch["part_mask"].bool()
        pid = batch["part_pid"][mask].to(torch.int64)
        counts += torch.bincount(pid, minlength=8).cpu()
        observed_events += int(mask.shape[0])
    if observed_events != events:
        raise RuntimeError(f"Expected {events} events, observed {observed_events}")
    if torch.any(counts == 0):
        raise RuntimeError(f"At least one PID class was absent: {counts.tolist()}")

    total = int(counts.sum())
    weights = total / (8.0 * counts.to(torch.float64))
    manifest_paths = {
        name: str(Path(str(spec["paths"])).resolve())
        for name, spec in data_cfg.train_val_processes.items()
    }
    return {
        "selection": selection,
        "experiment": SELECTION_EXPERIMENTS[selection],
        "events": observed_events,
        "particles": total,
        "pid_counts": counts.tolist(),
        "inverse_frequency_weights": [float(value) for value in weights],
        "normalization": "total_particles / (8 * pid_count)",
        "split_seed": int(data_cfg.split_seed),
        "shuffle_seed": int(data_cfg.shuffle_seed),
        "manifests": {
            name: {"path": path, "sha256": sha256(Path(path))}
            for name, path in manifest_paths.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.events < 1:
        raise ValueError("--events must be positive")
    os.environ["ORBIT_MANIFEST_DIR"] = str(args.manifest_dir.expanduser().resolve())
    os.environ.setdefault("LOG_DIR", str(REPO_ROOT / "logs"))
    selections = (
        tuple(SELECTION_EXPERIMENTS)
        if args.selection == "all"
        else (args.selection,)
    )
    output = {
        "event_sample": args.events,
        "top_level_training_groups": ["QCD", "tt", "VJets", "VV", "DY"],
        "selections": {
            selection: estimate(selection, args.events, args.num_workers)
            for selection in selections
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
