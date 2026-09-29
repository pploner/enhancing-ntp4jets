#!/usr/bin/env python3
"""Plot original and decoded particles for four representative ORBIT samples.

The particle selection is recovered from the tokenizer run configuration, so a
standard run produces the established ``PuppiW > 0.05``/128-particle display
and a full-event run produces the unfiltered/500-particle display.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gabbro.data.orbit_parquet import OrbitParquetDataset, SEQUENCE_SCHEMAS
from gabbro.plotting.orbit import plot_event_reconstruction_grid


SAMPLE_MANIFESTS = {
    "ggHbb": ["ggHbb_test.txt"],
    "ttbar": [
        "tt0123j_5f_ckm_LO_MLM_hadronic_test.txt",
        "tt0123j_5f_ckm_LO_MLM_leptonic_test.txt",
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt",
    ],
    "QCD": ["QCD_HT50tobb_test.txt", "QCD_HT50toInf_test.txt"],
    "Drell-Yan": ["DYJetsToLL_13TeV-madgraphMLM-pythia8_test.txt"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "production_final",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--events-per-sample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_model(run_dir: Path, device: torch.device):
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        config_path = run_dir / ".hydra" / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(f"No resolved or Hydra config found under {run_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), cfg, config_path, checkpoint_path


def checkpoint_data_settings(cfg) -> dict:
    sequence_type = str(cfg.data.get("sequence_type", "particle"))
    if sequence_type not in {"particle", "particle_full"}:
        raise ValueError(
            "Event reconstruction examples require a particle tokenizer, got "
            f"{sequence_type!r}"
        )
    schema = SEQUENCE_SCHEMAS[sequence_type]
    max_sequence_length = cfg.data.get("max_sequence_length")
    mask_column = cfg.data.get("mask_column")
    mask_min_value = cfg.data.get("mask_min_value")
    return {
        "sequence_type": sequence_type,
        "max_sequence_length": int(
            schema["max_sequence_length"]
            if max_sequence_length is None
            else max_sequence_length
        ),
        "mask_column": schema["mask_column"] if mask_column is None else str(mask_column),
        "mask_min_value": float(
            schema["mask_min_value"]
            if mask_min_value is None
            else mask_min_value
        ),
        "include_energy": bool(cfg.data.get("include_energy", False)),
        "energy_shift": float(cfg.data.get("energy_shift", 2.5)),
    }


def to_physical(values: np.ndarray) -> np.ndarray:
    """Convert [eta/3, cos(phi), sin(phi), log(pt)-1.8, ...] to eta, phi, pt."""
    return np.stack(
        [
            values[..., 0] * 3.0,
            np.arctan2(values[..., 2], values[..., 1]),
            np.clip(np.exp(values[..., 3] + 1.8) - 1e-8, 0.0, None),
        ],
        axis=-1,
    )


def decode_batch(model, batch: dict, device: torch.device) -> np.ndarray:
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
    return to_physical(reconstructed.detach().cpu().numpy())


def collect_sample(
    model,
    cfg,
    settings: dict,
    manifests: list[Path],
    events: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    dataset = OrbitParquetDataset(
        manifests,
        sequence_type=settings["sequence_type"],
        batch_size=events,
        max_sequence_length=settings["max_sequence_length"],
        shuffle_row_groups=True,
        shuffle_seed=seed,
        mask_column=settings["mask_column"],
        mask_min_value=settings["mask_min_value"],
        max_events=events,
        return_event_metadata=True,
        pid_cfg=cfg.get("pid"),
        include_energy=settings["include_energy"],
        energy_shift=settings["energy_shift"],
    )
    originals = []
    reconstructed = []
    masks = []
    sources = []
    for batch in dataset:
        batch_original = to_physical(batch["part_features"].numpy())
        batch_reconstructed = decode_batch(model, batch, device)
        batch_mask = batch["part_mask"].numpy().astype(bool)
        originals.append(batch_original)
        reconstructed.append(batch_reconstructed)
        masks.append(batch_mask)
        sources.extend(
            {
                "file": str(source_file),
                "row": int(source_row),
                "selected_particles": int(event_mask.sum()),
            }
            for source_file, source_row, event_mask in zip(
                batch["source_files"],
                batch["source_rows"],
                batch_mask,
                strict=True,
            )
        )
    if not originals:
        raise RuntimeError(f"No events loaded from {', '.join(map(str, manifests))}")
    return (
        np.concatenate(originals)[:events],
        np.concatenate(reconstructed)[:events],
        np.concatenate(masks)[:events],
        sources[:events],
    )


def main() -> None:
    args = parse_args()
    if args.events_per_sample < 1:
        raise ValueError("--events-per-sample must be positive")
    run_dir = args.run_dir.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output = args.output.resolve()
    device = torch.device(args.device)
    model, cfg, config_path, checkpoint_path = load_model(run_dir, device)
    settings = checkpoint_data_settings(cfg)

    event_groups = []
    selected_sources = {}
    for sample_index, (sample_name, manifest_names) in enumerate(SAMPLE_MANIFESTS.items()):
        manifests = [manifest_dir / name for name in manifest_names]
        missing = [str(path) for path in manifests if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing manifests:\n  " + "\n  ".join(missing))
        original, reconstructed, mask, sources = collect_sample(
            model,
            cfg,
            settings,
            manifests,
            args.events_per_sample,
            args.seed + sample_index,
            device,
        )
        event_groups.append((sample_name, original, reconstructed, mask))
        selected_sources[sample_name] = sources

    figure = plot_event_reconstruction_grid(
        event_groups, events_per_group=args.events_per_sample
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "selection": settings,
        "events_per_sample": args.events_per_sample,
        "seed": args.seed,
        "manifests": {
            sample: [str(manifest_dir / name) for name in names]
            for sample, names in SAMPLE_MANIFESTS.items()
        },
        "events": selected_sources,
        "output": str(output),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved event reconstruction grid to {output}")
    print(f"Saved event provenance to {metadata_path}")


if __name__ == "__main__":
    main()
