"""Paired original/decoded ORBIT datasets for downstream fidelity tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import awkward as ak
import lightning as L
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


from gabbro.data.orbit_taxonomy import (
    FIVE_CLASS_GROUPS,
    FIVE_CLASS_TO_LABEL,
    PROCESS_TO_GROUP,
)


def physical_particles_to_classifier_features(
    particles: ak.Array,
    pid: ak.Array,
    max_sequence_length: int,
    pid_num_classes: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ragged ``[eta, phi, pT]`` and PID to padded classifier features."""
    particles = particles[:, :max_sequence_length]
    pid = pid[:, :max_sequence_length]
    lengths = np.asarray(ak.to_numpy(ak.num(particles, axis=1)), dtype=np.int64)
    mask = torch.arange(max_sequence_length).unsqueeze(0) < torch.from_numpy(lengths).unsqueeze(1)

    padded = ak.fill_none(
        ak.pad_none(particles, max_sequence_length, axis=1, clip=True),
        [0.0, 0.0, 0.0],
        axis=1,
    )
    values = np.asarray(ak.to_numpy(padded), dtype=np.float32)
    # The classifier's own fixed input transform, deliberately independent of
    # the tokenizer's data.eta_scale/pt_shift: inputs here are already physical,
    # and a fixed transform keeps classifiers comparable across tokenizer runs.
    transformed = np.stack(
        [
            values[..., 0] / 3.0,
            np.cos(values[..., 1]),
            np.sin(values[..., 1]),
            np.log(np.clip(values[..., 2], 0.0, None) + 1e-8) - 1.8,
        ],
        axis=-1,
    )
    transformed[~mask.numpy()] = 0.0

    padded_pid = ak.fill_none(ak.pad_none(pid, max_sequence_length, axis=1, clip=True), -1)
    pid_values = np.asarray(ak.to_numpy(padded_pid), dtype=np.int64)
    pid_one_hot = np.zeros((*pid_values.shape, pid_num_classes), dtype=np.float32)
    valid_pid = (pid_values >= 0) & (pid_values < pid_num_classes) & mask.numpy()
    rows, columns = np.nonzero(valid_pid)
    pid_one_hot[rows, columns, pid_values[rows, columns]] = 1.0
    features = np.concatenate([transformed, pid_one_hot], axis=-1)
    return torch.from_numpy(features), mask


class PairedOrbitParquetDataset(IterableDataset):
    """Stream classifier batches from sharded paired-event parquet files."""

    def __init__(
        self,
        files: list[str | Path],
        representation: Literal["original", "decoded"],
        batch_size: int = 256,
        max_sequence_length: int = 128,
        shuffle_row_groups: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        if representation not in {"original", "decoded"}:
            raise ValueError("representation must be 'original' or 'decoded'")
        self.representation = representation
        self.batch_size = int(batch_size)
        self.max_sequence_length = int(max_sequence_length)
        self.shuffle_row_groups = bool(shuffle_row_groups)
        self.seed = int(seed)
        self._iteration = 0
        self.row_groups = []
        for path in sorted(map(str, files)):
            parquet = pq.ParquetFile(path)
            self.row_groups.extend((path, index) for index in range(parquet.num_row_groups))
        if not self.row_groups:
            raise ValueError("No paired parquet row groups were found")

    def __iter__(self):
        row_groups = list(self.row_groups)
        if self.shuffle_row_groups:
            np.random.default_rng(self.seed + self._iteration).shuffle(row_groups)
            self._iteration += 1
        worker = get_worker_info()
        if worker is not None:
            row_groups = row_groups[worker.id :: worker.num_workers]

        feature_column = f"{self.representation}_particles"
        pid_column = f"{self.representation}_pid"
        columns = [feature_column, pid_column, "class_label", "process_label", "event_id"]
        for path, row_group in row_groups:
            parquet = pq.ParquetFile(path)
            for record_batch in parquet.iter_batches(
                row_groups=[row_group], columns=columns, batch_size=self.batch_size
            ):
                records = ak.from_arrow(record_batch)
                features, mask = physical_particles_to_classifier_features(
                    records[feature_column], records[pid_column], self.max_sequence_length
                )
                yield {
                    "part_features": features,
                    "part_mask": mask,
                    "labels": torch.from_numpy(
                        np.asarray(ak.to_numpy(records.class_label), dtype=np.int64)
                    ),
                    "process_labels": torch.from_numpy(
                        np.asarray(ak.to_numpy(records.process_label), dtype=np.int64)
                    ),
                    "event_ids": np.asarray(ak.to_list(records.event_id), dtype=object),
                }


class PairedOrbitDataModule(L.LightningDataModule):
    """Five-class paired data with original/decoded test loaders."""

    def __init__(
        self,
        data_dir: str,
        train_representation: Literal["original", "decoded"] = "original",
        batch_size: int | None = None,
        num_workers: int = 4,
        max_sequence_length: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        metadata_path = Path(data_dir) / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata.get("class_to_label") != FIVE_CLASS_TO_LABEL:
            raise ValueError("Paired data class mapping does not match the canonical mapping")
        source_length = int(self.metadata.get("max_sequence_length", 128))
        if max_sequence_length is not None and int(max_sequence_length) != source_length:
            raise ValueError(
                "Configured classifier length does not match paired data: "
                f"{max_sequence_length} != {source_length}"
            )
        self.hparams.max_sequence_length = source_length
        self.hparams.batch_size = (
            int(batch_size) if batch_size is not None else (16 if source_length > 128 else 256)
        )

    def _files(self, split: str) -> list[Path]:
        files = sorted((Path(self.hparams.data_dir) / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No paired parquet shards for split {split!r}")
        return files

    def _loader(self, split: str, representation: str, shuffle: bool) -> DataLoader:
        dataset = PairedOrbitParquetDataset(
            self._files(split),
            representation=representation,
            batch_size=self.hparams.batch_size,
            max_sequence_length=self.hparams.max_sequence_length,
            shuffle_row_groups=shuffle,
            seed=self.hparams.seed,
        )
        kwargs = {
            "batch_size": None,
            "num_workers": self.hparams.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": self.hparams.num_workers > 0,
        }
        if self.hparams.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._loader("train", self.hparams.train_representation, shuffle=True)

    def val_dataloader(self):
        return self._loader("val", self.hparams.train_representation, shuffle=False)

    def test_dataloader(self):
        return [
            self._loader("test", "original", shuffle=False),
            self._loader("test", "decoded", shuffle=False),
        ]
