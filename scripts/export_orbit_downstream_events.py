#!/usr/bin/env python
"""Export balanced, paired original/decoded events for downstream classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from gabbro.data.orbit_transforms import KinematicTransform
from gabbro.data.orbit_downstream import (
    FIVE_CLASS_GROUPS,
    FIVE_CLASS_TO_LABEL,
    PROCESS_TO_GROUP,
)
from gabbro.data.orbit_parquet import (
    SEQUENCE_SCHEMAS,
    CanonicalOrbitParquetDataModule,
    _dataset_files,
)


DEFAULT_SPLIT_BUDGETS = {"train": 200_000, "val": 200_000, "test": 20_000}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--events-per-shard", type=int, default=10_000)
    parser.add_argument(
        "--train-events",
        type=int,
        default=DEFAULT_SPLIT_BUDGETS["train"],
        help="Balanced total number of training events to export.",
    )
    parser.add_argument(
        "--val-events",
        type=int,
        default=DEFAULT_SPLIT_BUDGETS["val"],
        help="Balanced total number of validation events to export.",
    )
    parser.add_argument(
        "--test-events",
        type=int,
        default=DEFAULT_SPLIT_BUDGETS["test"],
        help="Balanced total number of test events to export.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--allow-no-pid",
        action="store_true",
        help="Export a kinematics-only paired dataset for a checkpoint without PID inputs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(run_dir: Path, device: torch.device, allow_no_pid: bool):
    config_path = run_dir / "config_resolved.yaml"
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = OmegaConf.load(config_path)
    pid_enabled = bool(cfg.get("pid", {}).get("enabled", False))
    if not pid_enabled and not allow_no_pid:
        raise ValueError(
            "The canonical downstream classifier requires a PID-enabled checkpoint. "
            "Pass --allow-no-pid to export a clearly labeled kinematics-only dataset."
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
    model = instantiate(cfg.model)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), cfg, config_path, checkpoint, pid_enabled


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


def build_data_module(
    manifest_dir: Path,
    batch_size: int,
    sequence_type: str,
    max_sequence_length: int,
    mask_column: str | None,
    mask_min_value: float,
    pid_enabled: bool,
    split_budgets: dict[str, int],
    kinematics: KinematicTransform,
):
    train_specs = {}
    test_specs = {}
    train_files = set()
    test_files = set()
    manifest_hashes = {}
    for group, processes in FIVE_CLASS_GROUPS.items():
        for process in processes:
            train_manifest = manifest_dir / f"{process}_train_val.txt"
            test_manifest = manifest_dir / f"{process}_test.txt"
            if not train_manifest.is_file():
                raise FileNotFoundError(train_manifest)
            if not test_manifest.is_file():
                raise FileNotFoundError(test_manifest)
            train_specs[process] = {"paths": str(train_manifest), "group": group}
            test_specs[process] = {"paths": str(test_manifest), "group": group}
            process_train_files = set(_dataset_files(train_manifest))
            process_test_files = set(_dataset_files(test_manifest))
            overlap = process_train_files & process_test_files
            if overlap:
                raise ValueError(f"Train/test manifest overlap for {process}: {sorted(overlap)[:3]}")
            train_files.update(process_train_files)
            test_files.update(process_test_files)
            manifest_hashes[train_manifest.name] = sha256(train_manifest)
            manifest_hashes[test_manifest.name] = sha256(test_manifest)
    if train_files & test_files:
        raise ValueError("Global train/test parquet overlap detected")

    module = CanonicalOrbitParquetDataModule(
        train_val_processes=train_specs,
        test_suites={
            "classification": {"event_budget": split_budgets["test"], "processes": test_specs}
        },
        train_event_budget=split_budgets["train"],
        val_event_budget=split_budgets["val"],
        sequence_type=sequence_type,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        mask_column=mask_column,
        mask_min_value=mask_min_value,
        num_workers=0,
        pid_cfg={"enabled": pid_enabled, "num_classes": 8},
        return_raw_features=True,
        return_event_metadata=True,
        **kinematics.as_kwargs(),
    )
    if dict(module.hparams.group_to_label) != FIVE_CLASS_TO_LABEL:
        raise ValueError("Canonical loader generated an unexpected class-label mapping")
    return module, manifest_hashes


def decode_batch(model, batch, device, pid_enabled: bool):
    features = batch["part_features"].to(device)
    mask = batch["part_mask"].to(device)
    pid = batch.get("part_pid")
    if pid is not None:
        pid = pid.to(device)
    with torch.inference_mode():
        if hasattr(model, "_quantize_batch"):
            reconstructed, _, pid_logits = model._quantize_batch(features, mask, pid)
        else:
            reconstructed, outputs = model.forward(features, mask, pid_particle=pid)
            pid_logits = outputs.get("pid_logits")
    if pid_enabled and pid_logits is None:
        raise ValueError("PID-enabled checkpoint did not return decoded PID logits")
    decoded_pid = (
        pid_logits.argmax(dim=-1).detach().cpu().numpy()
        if pid_logits is not None
        else np.full(mask.shape, -1, dtype=np.int64)
    )
    return (
        reconstructed.detach().cpu().numpy(),
        decoded_pid,
        mask.detach().cpu().numpy().astype(bool),
    )


def transformed_to_physical(values: np.ndarray, transform: KinematicTransform) -> np.ndarray:
    """Undo the run's kinematic preprocessing: ``[..., >=4]`` -> ``[eta, phi, pT]``."""
    return transform.to_physical(values)


def prepare_test_time_baseline(model, data_module, output_dir: Path):
    """Fit stateful learned-bin/FAISS baselines before streaming exports."""
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
    if getattr(model, "binning", None) == "learned" and hasattr(model, "_fit_learned_bins"):
        model._fit_learned_bins()
    if hasattr(model, "_fit_faiss"):
        model._fit_faiss()


def write_shard(split_dir: Path, shard_index: int, rows: dict[str, list]):
    path = split_dir / f"events-{shard_index:05d}.parquet"
    table = pa.table({name: pa.array(values) for name, values in rows.items()})
    pq.write_table(table, path, compression="zstd", row_group_size=2048)
    return path


def export_split(
    model, loader, split, output_dir, device, budget, events_per_shard, pid_enabled, kinematics
):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=False)
    columns = {
        "event_id": [],
        "source_file": [],
        "source_row": [],
        "class_label": [],
        "process_label": [],
        "original_particles": [],
        "decoded_particles": [],
        "original_pid": [],
        "decoded_pid": [],
    }
    shard_index = 0
    exported = 0
    shards = []
    with tqdm(total=budget, desc=f"Exporting {split}", unit="event") as progress:
        for batch in loader:
            reconstructed, decoded_pid, mask = decode_batch(model, batch, device, pid_enabled)
            decoded_physical = transformed_to_physical(reconstructed, kinematics)
            raw = batch["raw_part_features"]
            source_rows = batch["source_rows"].numpy()
            labels = batch["jet_type_labels"].numpy()
            process_labels = batch["process_labels"].numpy()
            original_pid = (
                batch["part_pid"].numpy()
                if pid_enabled
                else np.full(mask.shape, -1, dtype=np.int64)
            )
            for index in range(mask.shape[0]):
                valid = mask[index]
                source_file = str(batch["source_files"][index])
                source_row = int(source_rows[index])
                event_id = hashlib.sha256(
                    f"{source_file}:{source_row}".encode("utf-8")
                ).hexdigest()
                raw_event = np.asarray(ak.to_numpy(raw[index]), dtype=np.float32)
                # OrbitParquetDataset raw order is [pT, eta, phi].
                original = raw_event[:, [1, 2, 0]]
                columns["event_id"].append(event_id)
                columns["source_file"].append(source_file)
                columns["source_row"].append(source_row)
                columns["class_label"].append(int(labels[index]))
                columns["process_label"].append(int(process_labels[index]))
                columns["original_particles"].append(original.tolist())
                columns["decoded_particles"].append(decoded_physical[index, valid].tolist())
                columns["original_pid"].append(original_pid[index, valid].tolist())
                columns["decoded_pid"].append(decoded_pid[index, valid].tolist())
                exported += 1
                if len(columns["event_id"]) == events_per_shard:
                    shards.append(str(write_shard(split_dir, shard_index, columns)))
                    shard_index += 1
                    columns = {name: [] for name in columns}
            progress.update(mask.shape[0])
    if columns["event_id"]:
        shards.append(str(write_shard(split_dir, shard_index, columns)))
    if exported != budget:
        raise RuntimeError(f"Expected {budget} {split} events, exported {exported}")
    return {"events": exported, "shards": shards}


def main():
    args = parse_args()
    split_budgets = {
        "train": args.train_events,
        "val": args.val_events,
        "test": args.test_events,
    }
    if any(budget <= 0 for budget in split_budgets.values()):
        raise ValueError(f"All split budgets must be positive: {split_budgets}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, config_path, checkpoint, pid_enabled = load_model(
        args.run_dir.resolve(), device, args.allow_no_pid
    )
    sequence_type, max_sequence_length, mask_column, mask_min_value = (
        checkpoint_data_settings(cfg)
    )
    batch_size = args.batch_size or (16 if max_sequence_length > 128 else 256)
    kinematics = KinematicTransform.from_data_config(cfg.data)
    data_module, manifest_hashes = build_data_module(
        args.manifest_dir.resolve(),
        batch_size,
        sequence_type,
        max_sequence_length,
        mask_column,
        mask_min_value,
        pid_enabled,
        split_budgets,
        kinematics,
    )
    prepare_test_time_baseline(model, data_module, args.output_dir)
    loaders = {
        "train": data_module.train_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader()[0],
    }
    split_outputs = {
        split: export_split(
            model,
            loader,
            split,
            args.output_dir,
            device,
            split_budgets[split],
            args.events_per_shard,
            pid_enabled,
            kinematics,
        )
        for split, loader in loaders.items()
    }
    metadata = {
        "format": "orbit-paired-downstream",
        "version": 1,
        "class_to_label": FIVE_CLASS_TO_LABEL,
        "process_to_group": PROCESS_TO_GROUP,
        "process_to_label": dict(data_module.hparams.process_to_label),
        "split_budgets": split_budgets,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path),
        "manifest_hashes": manifest_hashes,
        "quota_summary": data_module.data_split_summary(),
        "pid_enabled": pid_enabled,
        "pid_num_classes": 8,
        "particle_features": ["eta", "phi", "pt"],
        "sequence_type": sequence_type,
        "max_sequence_length": max_sequence_length,
        "selection": {
            "mask_column": mask_column,
            "mask_min_value": mask_min_value,
        },
        "splits": split_outputs,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote paired downstream data to {args.output_dir}")


if __name__ == "__main__":
    main()
