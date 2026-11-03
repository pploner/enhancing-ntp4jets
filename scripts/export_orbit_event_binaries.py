#!/usr/bin/env python
"""Export original model inputs and quantized token IDs for ORBIT test events."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Prefer this checkout and its vendored vqtorch over stale editable installs in
# the user or Condor environment. This must happen before importing the model.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    import_root_string = str(import_root)
    if import_root_string in sys.path:
        sys.path.remove(import_root_string)
    sys.path.insert(0, import_root_string)

import awkward as ak
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

try:
    import pyrootutils

    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except ImportError:
    pass

from gabbro.data.orbit_parquet import OrbitParquetDataset
from gabbro.utils.orbit_binary import unsigned_dtype_for_size, write_ragged_binary


DEFAULT_CLASSES = ("ggHbb", "minbias")
RAW_FEATURE_SUFFIXES = ("PT", "Eta", "Phi")


def _quantizer_shape(cfg) -> tuple[int, int]:
    """Return ``(per-stage codebook size, number of indices per latent)``."""
    residual_cfg = cfg.model.model_kwargs.get("residual_quantizer_cfg")
    if residual_cfg:
        return int(residual_cfg.codebook_size), int(residual_cfg.num_quantizers)
    return int(cfg.model.model_kwargs.vq_kwargs.num_codes), 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Hydra run directory containing config_resolved.yaml and checkpoints/best.ckpt.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events-per-class", type=int, default=1000)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Inference batch size (default: 16 above length 128, otherwise 256).",
    )
    parser.add_argument(
        "--disable-event-filter",
        action="store_true",
        help=(
            "Ignore class-level eval_sequence_type/eval_min_pt selection from the run config. "
            "Particle-level selections, including the PUPPI mask, are unchanged."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference (default: cuda when available).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(cfg, checkpoint: Path, device: torch.device):
    import vqtorch

    vendored_vqtorch = VQTORCH_ROOT.resolve()
    loaded_vqtorch = Path(vqtorch.__file__).resolve()
    if vendored_vqtorch not in loaded_vqtorch.parents:
        raise RuntimeError(
            "Expected the repository's vendored vqtorch, but imported "
            f"{loaded_vqtorch}. Expected it below {vendored_vqtorch}."
        )
    print(f"Using vqtorch from {loaded_vqtorch}")

    # The checkpoint is produced by this repository and contains Lightning
    # hyperparameters such as functools.partial, which PyTorch 2.6's restricted
    # weights-only unpickler intentionally rejects.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
    if "state_dict" not in checkpoint_data:
        raise KeyError(f"Lightning checkpoint has no state_dict: {checkpoint}")
    model = instantiate(cfg.model)
    model.load_state_dict(checkpoint_data["state_dict"], strict=True)
    return model.to(device).eval()


def _class_dataset(
    cfg,
    class_name: str,
    events: int,
    batch_size: int,
    disable_event_filter: bool = False,
):
    all_specs = cfg.data.get("parquet_files_test_per_class")
    if all_specs is not None and class_name in all_specs:
        spec = all_specs[class_name]
    else:
        matching_specs = []
        for suite in (cfg.data.get("test_suites") or {}).values():
            processes = suite.get("processes", {})
            if class_name in processes:
                matching_specs.append(processes[class_name])
        if not matching_specs:
            raise KeyError(
                f"Process/class {class_name!r} is absent from the configured test data"
            )
        spec = matching_specs[0]
    sequence_type = str(spec.get("sequence_type", cfg.data.sequence_type))
    paths = spec.get("paths", spec.get("files", spec.get("manifest")))
    if paths is None:
        raise KeyError(f"Test specification for {class_name!r} has no paths")
    return OrbitParquetDataset(
        paths,
        sequence_type=sequence_type,
        max_sequence_length=cfg.data.get("max_sequence_length"),
        batch_size=batch_size,
        shuffle_row_groups=False,
        mask_column=cfg.data.get("mask_column"),
        mask_min_value=cfg.data.get("mask_min_value"),
        min_pt=spec.get("min_pt", cfg.data.get("min_pt")),
        event_filter_sequence_type=(
            None if disable_event_filter else spec.get("eval_sequence_type")
        ),
        event_filter_min_pt=None if disable_event_filter else spec.get("eval_min_pt"),
        max_events=events,
        return_raw_features=True,
        pid_cfg=cfg.get("pid"),
    )


def _collect_class(
    model,
    dataset,
    device: torch.device,
    expected_events: int,
    num_codes: int,
    progress_description: str,
    num_quantizers: int = 1,
):
    originals: list[np.ndarray] = []
    pid_events: list[np.ndarray] = []
    token_events: list[np.ndarray] = []
    with tqdm(total=expected_events, desc=progress_description, unit="event") as progress:
        with torch.inference_mode():
            for batch in dataset:
                features = batch["part_features"].to(device)
                particle_mask = batch["part_mask"].to(device)
                particle_pid = batch.get("part_pid")
                if particle_pid is not None:
                    particle_pid = particle_pid.to(device)
                raw_features = batch["raw_part_features"]
                _, vq_out = model(
                    features,
                    particle_mask,
                    pid_particle=particle_pid,
                )
                codes = vq_out["q"]
                latent_mask = vq_out.get("latent_mask", particle_mask).bool()
                if codes.ndim == latent_mask.ndim + 1 and codes.shape[-1] == 1:
                    codes = codes.squeeze(-1)
                expected_code_shape = (
                    (*latent_mask.shape, num_quantizers)
                    if num_quantizers > 1
                    else latent_mask.shape
                )
                if codes.shape != expected_code_shape:
                    raise ValueError(
                        "Token tensor does not match the configured residual depth; "
                        f"got codes={tuple(codes.shape)}, expected={tuple(expected_code_shape)}"
                    )

                features_np = features.detach().cpu().numpy()
                particle_mask_np = particle_mask.detach().cpu().numpy().astype(bool)
                codes_np = codes.detach().cpu().numpy()
                latent_mask_np = latent_mask.detach().cpu().numpy().astype(bool)
                for event_idx in range(features_np.shape[0]):
                    event_codes = codes_np[event_idx][latent_mask_np[event_idx]].reshape(-1)
                    if np.any(event_codes < 0) or np.any(event_codes >= num_codes):
                        raise ValueError(
                            f"Token IDs must be in [0, {num_codes}); "
                            f"observed [{event_codes.min()}, {event_codes.max()}]"
                        )
                    raw_event = np.asarray(ak.to_numpy(raw_features[event_idx]))
                    expected_particles = int(particle_mask_np[event_idx].sum())
                    if raw_event.shape != (expected_particles, len(RAW_FEATURE_SUFFIXES)):
                        raise ValueError(
                            "Raw parquet and model particle selections differ: "
                            f"raw={raw_event.shape}, expected=({expected_particles}, "
                            f"{len(RAW_FEATURE_SUFFIXES)})"
                        )
                    originals.append(raw_event)
                    if particle_pid is not None:
                        pid_events.append(
                            particle_pid[event_idx][particle_mask[event_idx]]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8, copy=False)
                        )
                    token_events.append(event_codes)
                progress.update(features_np.shape[0])

    if len(originals) != expected_events:
        raise RuntimeError(
            f"Requested {expected_events} events but the filtered dataset yielded {len(originals)}"
        )
    return originals, pid_events, token_events


def main() -> None:
    args = parse_args()
    if args.events_per_class < 1:
        raise ValueError("--events-per-class must be positive")

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config_resolved.yaml"
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    cfg = OmegaConf.load(config_path)
    configured_length = cfg.data.get("max_sequence_length")
    sequence_type = str(cfg.data.get("sequence_type", "particle"))
    default_lengths = {"particle": 128, "particle_full": 500}
    effective_length = (
        int(configured_length)
        if configured_length is not None
        else default_lengths.get(sequence_type, 128)
    )
    batch_size = args.batch_size or (16 if effective_length > 128 else 256)
    num_codes, num_quantizers = _quantizer_shape(cfg)
    token_dtype = unsigned_dtype_for_size(num_codes)
    device = torch.device(args.device)
    model = _load_model(cfg, checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "orbit-event-binary-export",
        "version": 1,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "device": str(device),
        "events_per_class": args.events_per_class,
        "event_filter_disabled": args.disable_event_filter,
        "num_codes": num_codes,
        "num_quantizers": num_quantizers,
        "total_token_combinations": num_codes**num_quantizers,
        "stored_code_vectors": num_codes * num_quantizers,
        "fixed_bits_per_latent": int(num_codes - 1).bit_length() * num_quantizers,
        "classes": {},
    }
    for class_name in args.classes:
        dataset = _class_dataset(
            cfg,
            class_name,
            args.events_per_class,
            batch_size,
            disable_event_filter=args.disable_event_filter,
        )
        originals, pids, tokens = _collect_class(
            model,
            dataset,
            device,
            args.events_per_class,
            num_codes,
            progress_description=f"Exporting {class_name}",
            num_quantizers=num_quantizers,
        )
        original_path = args.output_dir / f"{class_name}_original_parquet.bin"
        token_path = args.output_dir / f"{class_name}_vq_tokens.bin"
        pid_path = args.output_dir / f"{class_name}_pid.bin"
        common = {
            "class_name": class_name,
            "source_split": "test",
            "checkpoint": str(checkpoint),
        }
        write_ragged_binary(
            original_path,
            originals,
            dtype=originals[0].dtype,
            width=len(RAW_FEATURE_SUFFIXES),
            metadata={
                **common,
                "kind": "original_parquet_particles",
                "feature_names": list(dataset.raw_output_features),
                "preprocessing": "none",
                "particle_selection": "same mask, order, and truncation as tokenizer",
            },
        )
        if pids:
            write_ragged_binary(
                pid_path,
                pids,
                dtype=np.dtype("uint8"),
                metadata={
                    **common,
                    "kind": "mapped_particle_pid",
                    "class_names": list(cfg.pid.class_names),
                    "particle_selection": "same mask, order, and truncation as tokenizer",
                },
            )
        write_ragged_binary(
            token_path,
            tokens,
            dtype=token_dtype,
            metadata={
                **common,
                "kind": "quantized_token_ids",
                "num_codes": num_codes,
                "num_quantizers": num_quantizers,
                "token_order": "latent-major, then residual stage",
                "fixed_bits_per_latent": (
                    int(num_codes - 1).bit_length() * num_quantizers
                ),
                "latent_sequence_ratio": float(
                    cfg.model.model_kwargs.get("latent_sequence_cfg", {}).get("ratio", 1.0)
                    if cfg.model.model_kwargs.get("latent_sequence_cfg")
                    else 1.0
                ),
            },
        )
        manifest["classes"][class_name] = {
            "selection": {
                "sequence_type": dataset.sequence_type,
                "max_sequence_length": dataset.max_sequence_length,
                "mask_column": dataset.mask_column,
                "mask_min_value": dataset.mask_min_value,
                "event_filter_pt_column": dataset.event_filter_pt_column,
                "event_filter_min_pt": dataset.event_filter_min_pt,
            },
            "pid": (
                {
                    "path": pid_path.name,
                    "bytes": pid_path.stat().st_size,
                    "sha256": _sha256(pid_path),
                    "dtype": "uint8",
                    "class_names": list(cfg.pid.class_names),
                }
                if pids
                else None
            ),
            "original": {
                "path": original_path.name,
                "bytes": original_path.stat().st_size,
                "sha256": _sha256(original_path),
            },
            "tokens": {
                "path": token_path.name,
                "bytes": token_path.stat().st_size,
                "sha256": _sha256(token_path),
            },
            "input_particles": int(sum(len(event) for event in originals)),
            "latent_tokens": int(sum(len(event) for event in tokens)),
        }
        print(
            f"{class_name}: {len(originals)} events, "
            f"{original_path.stat().st_size} original bytes, "
            f"{token_path.stat().st_size} token bytes"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
