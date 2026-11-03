"""Iterable parquet loaders for ORBIT particle and jet sequences.

Source-agnostic pieces (schemas, preprocessing, batch conversion, quotas) live
in :mod:`gabbro.data.orbit_common`; the canonical data module lives in
:mod:`gabbro.data.orbit_canonical`. Both are re-exported here so existing
imports and saved run configs keep working.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import awkward as ak
import lightning as L
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import ChainDataset, DataLoader

from gabbro.data.orbit_common import (  # noqa: F401  (re-exported)
    MANIFEST_SUFFIXES,
    PID_CLASS_NAMES,
    SEQUENCE_SCHEMAS,
    OrbitPreprocessor,
    OrbitSequenceDataset,
    WeightedClassDataset,
    _log_worker_exception,
    balanced_group_process_quotas,
    expand_paths,
    map_pdg_charge_to_pid_class,
    resolve_process_quotas,
)

PARQUET_SUFFIXES = {".parquet"}


def _as_paths(paths) -> list[str]:
    return expand_paths(paths, PARQUET_SUFFIXES)


def _dataset_files(paths) -> list[str]:
    paths = _as_paths(paths)
    if not paths:
        return []
    files = []
    for path in paths:
        # Manifests normally expand to explicit parquet files. Passing every
        # one back through pyarrow.dataset needlessly discovers it again; only
        # directories and other dataset-like paths require Arrow discovery.
        if Path(path).suffix.lower() in PARQUET_SUFFIXES:
            files.append(str(path))
        else:
            files.extend(ds.dataset(path, format="parquet").files)
    return sorted(set(files))


def _limit_files(files: list[str], max_files: Optional[int]) -> list[str]:
    if max_files is None:
        return files
    if max_files < 1:
        raise ValueError("max_files_per_class values must be positive")
    return files[:max_files]


def deterministic_file_split(
    parquet_files,
    train_fraction: float = 0.8,
    split_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Shuffle parquet files deterministically and return disjoint train and val lists."""
    files = _dataset_files(parquet_files)
    if len(files) < 2:
        raise ValueError(
            "Automatic train/val splitting requires at least two parquet files. "
            "For a smoke test, provide parquet_files_train and parquet_files_val explicitly."
        )
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    rng = np.random.default_rng(split_seed)
    files = [str(path) for path in np.asarray(files)[rng.permutation(len(files))]]
    split_index = min(max(int(len(files) * train_fraction), 1), len(files) - 1)
    return files[:split_index], files[split_index:]


class OrbitParquetDataset(OrbitSequenceDataset):
    """Stream pre-batched, padded ORBIT events from parquet row groups."""

    source_kind = "parquet"

    def __init__(self, parquet_files, *, shuffle_row_groups: bool = False, **kwargs):
        super().__init__(shuffle=shuffle_row_groups, **kwargs)
        parquet_files = _dataset_files(parquet_files)
        if not parquet_files:
            raise ValueError("parquet_files must contain at least one parquet file or directory")
        # Keep construction cheap.  In particular, do not open every parquet
        # file here to enumerate row groups: canonical datasets can contain
        # thousands of files while an event quota may consume only a handful.
        # Row groups are discovered lazily by each iterator and file scanning
        # stops as soon as max_events is reached.
        self.parquet_files = parquet_files

    def _iter_source_batches(self, worker_info, rng):
        parquet_files = list(self.parquet_files)
        if rng is not None:
            rng.shuffle(parquet_files)

        row_group_position = 0
        for file_path in parquet_files:
            batch_idx = None
            row_group_idx = None
            try:
                parquet_file = pq.ParquetFile(file_path)
                row_groups = []
                file_row_offset = 0
                for row_group_idx in range(parquet_file.num_row_groups):
                    row_groups.append((row_group_idx, file_row_offset))
                    file_row_offset += parquet_file.metadata.row_group(row_group_idx).num_rows
                if rng is not None:
                    rng.shuffle(row_groups)

                for row_group_idx, row_group_row_offset in row_groups:
                    assigned_to_worker = (
                        worker_info is None
                        or row_group_position % worker_info.num_workers == worker_info.id
                    )
                    row_group_position += 1
                    if not assigned_to_worker:
                        continue

                    batches = parquet_file.iter_batches(
                        row_groups=[row_group_idx],
                        columns=self.features,
                        batch_size=self.batch_size,
                        use_threads=True,
                    )
                    batch_row_offset = 0
                    for batch_idx, batch in enumerate(batches):
                        batch_rows = np.arange(
                            row_group_row_offset + batch_row_offset,
                            row_group_row_offset + batch_row_offset + batch.num_rows,
                            dtype=np.int64,
                        )
                        batch_row_offset += batch.num_rows
                        yield ak.from_arrow(batch), file_path, batch_rows
            except Exception:
                _log_worker_exception(
                    "failed while reading parquet "
                    f"file={file_path!r} row_group={row_group_idx} "
                    f"{self._describe()} last_batch_idx={batch_idx}",
                    self.source_kind,
                )
                raise


class OrbitParquetDataModule(L.LightningDataModule):
    """Separate Lightning data module for ORBIT particle or jet parquet sequences.

    Two modes for train/val data:

    Single-class mode (original):
        Pass ``parquet_files_train_val`` (auto-split) or explicit
        ``parquet_files_train`` / ``parquet_files_val``.  All samples receive
        ``jet_type_label`` as their class index.

    Multi-class mode:
        Pass ``parquet_files_train_val_per_class`` — a mapping of
        ``{class_name: paths_or_manifest}`` or
        ``{class_name: {paths, weight, eval_sequence_type, eval_min_pt}}``.
        Class names are sorted alphabetically to assign deterministic integer labels (0, 1, …).
        Each class is split independently (class-aware train/val split), then
        the resulting datasets are chained.  The ``class_to_label`` mapping is
        recorded in ``hparams`` for reproducibility.
    """

    def __init__(
        self,
        parquet_files_train_val=None,
        parquet_files_train_val_per_class: Optional[dict] = None,
        parquet_files_test_per_class: Optional[dict] = None,
        parquet_files_test=None,
        parquet_files_train=None,
        parquet_files_val=None,
        sequence_type: str = "particle",
        max_sequence_length: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        train_fraction: float = 0.8,
        split_seed: int = 42,
        shuffle_train: bool = True,
        shuffle_seed: int = 42,
        mask_column: Optional[str] = None,
        mask_min_value: Optional[float] = None,
        jet_type_label: int = 0,
        min_pt: Optional[float] = None,
        class_sequence_types: Optional[dict] = None,
        class_min_pts: Optional[dict] = None,
        class_eval_sequence_types: Optional[dict] = None,
        class_eval_min_pts: Optional[dict] = None,
        class_sampling_weights: Optional[dict] = None,
        max_files_per_class: Optional[dict] = None,
        max_train_events_per_class: Optional[dict] = None,
        max_val_events_per_class: Optional[dict] = None,
        max_test_events_per_class: Optional[dict] = None,
        pid_cfg: Optional[dict] = None,
        include_energy: bool = False,
        energy_shift: float = 2.5,
        auxiliary_jet_sequence_type: Optional[str] = None,
        max_auxiliary_jets: int = 5,
        **kwargs,
    ):
        super().__init__()
        if sequence_type not in SEQUENCE_SCHEMAS:
            raise ValueError(
                f"Unknown sequence_type {sequence_type!r}. "
                f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
            )

        if isinstance(batch_size, int):
            self.batch_size_train = batch_size
            self.batch_size_val = batch_size
            self.batch_size_test = batch_size
        else:
            required_splits = {"train", "val", "test"}
            if not required_splits.issubset(batch_size):
                raise ValueError(
                    "If batch_size is a mapping, it must include train, val, and test"
                )
            self.batch_size_train = batch_size["train"]
            self.batch_size_val = batch_size["val"]
            self.batch_size_test = batch_size["test"]

        if parquet_files_train_val_per_class:
            if parquet_files_train_val or parquet_files_train or parquet_files_val:
                raise ValueError(
                    "Use either parquet_files_train_val_per_class or the single-class "
                    "params (parquet_files_train_val / parquet_files_train+val), not both."
                )
            class_names = sorted(parquet_files_train_val_per_class.keys(), key=str.lower)
            self._class_to_label: dict[str, int] = {name: i for i, name in enumerate(class_names)}
            self._train_files_per_class: dict[str, list[str]] = {}
            self._val_files_per_class: dict[str, list[str]] = {}
            self._test_files_per_class: dict[str, list[str]] = {}
            self._class_specs = self._normalize_class_specs(
                parquet_files_train_val_per_class,
                sequence_type=sequence_type,
                min_pt=min_pt,
                class_sequence_types=class_sequence_types,
                class_min_pts=class_min_pts,
                class_eval_sequence_types=class_eval_sequence_types,
                class_eval_min_pts=class_eval_min_pts,
                class_sampling_weights=class_sampling_weights,
                max_files_per_class=max_files_per_class,
                max_train_events_per_class=max_train_events_per_class,
                max_val_events_per_class=max_val_events_per_class,
                max_test_events_per_class=max_test_events_per_class,
            )
            self._test_class_specs = self._normalize_class_specs(
                parquet_files_test_per_class or {},
                sequence_type=sequence_type,
                min_pt=min_pt,
                class_sequence_types=class_sequence_types,
                class_min_pts=class_min_pts,
                class_eval_sequence_types=class_eval_sequence_types,
                class_eval_min_pts=class_eval_min_pts,
                class_sampling_weights=class_sampling_weights,
                max_files_per_class=max_files_per_class,
                max_train_events_per_class=max_train_events_per_class,
                max_val_events_per_class=max_val_events_per_class,
                max_test_events_per_class=max_test_events_per_class,
            )
            if parquet_files_test_per_class:
                if sorted(parquet_files_test_per_class.keys(), key=str.lower) != class_names:
                    raise ValueError(
                        "parquet_files_test_per_class must contain the same class names "
                        "as parquet_files_train_val_per_class"
                    )
                if parquet_files_test:
                    raise ValueError(
                        "Use either parquet_files_test_per_class or parquet_files_test, not both."
                    )
            for name in class_names:
                train_f, val_f = deterministic_file_split(
                    self._class_specs[name]["paths"],
                    train_fraction=train_fraction,
                    split_seed=split_seed,
                )
                self._train_files_per_class[name] = _limit_files(
                    train_f, self._class_specs[name]["max_files"]
                )
                self._val_files_per_class[name] = _limit_files(
                    val_f, self._class_specs[name]["max_files"]
                )
                if parquet_files_test_per_class:
                    self._test_files_per_class[name] = _limit_files(
                        _dataset_files(self._test_class_specs[name]["paths"]),
                        self._test_class_specs[name]["max_files"],
                    )
            self.parquet_files_train = None
            self.parquet_files_val = None
            self._multi_class = True
        else:
            if parquet_files_train_val:
                if parquet_files_train or parquet_files_val:
                    raise ValueError(
                        "Use either parquet_files_train_val or explicit "
                        "parquet_files_train/parquet_files_val, not both."
                    )
                parquet_files_train, parquet_files_val = deterministic_file_split(
                    parquet_files_train_val,
                    train_fraction=train_fraction,
                    split_seed=split_seed,
                )
            if not parquet_files_train or not parquet_files_val:
                raise ValueError(
                    "Provide parquet_files_train_val, parquet_files_train_val_per_class, "
                    "or explicit parquet_files_train and parquet_files_val."
                )
            self.parquet_files_train = parquet_files_train
            self.parquet_files_val = parquet_files_val
            self._class_to_label = {}
            self._class_specs = {}
            self._test_class_specs = {}
            self._test_files_per_class = {}
            self._multi_class = False

        self.parquet_files_test = parquet_files_test
        self._max_sequence_length = (
            max_sequence_length
            if max_sequence_length is not None
            else SEQUENCE_SCHEMAS[sequence_type]["max_sequence_length"]
        )
        self.selected_features = OrbitPreprocessor(
            SEQUENCE_SCHEMAS[sequence_type]["prefix"],
            include_energy=include_energy,
            energy_shift=energy_shift,
        ).output_features
        self.save_hyperparameters(ignore=["parquet_files_train", "parquet_files_val"])
        self.hparams["selected_features"] = self.selected_features
        self.hparams["max_sequence_length"] = self._max_sequence_length
        if self._multi_class:
            self.hparams["class_to_label"] = self._class_to_label
            self.hparams["class_specs"] = self._class_specs

    def data_split_summary(self) -> dict:
        """Return resolved split metadata for logging and reproducibility."""

        def split_row(
            split: str,
            class_name: str,
            label: int,
            files: list[str],
            spec: dict,
            event_count: Optional[int],
            batch_size: int,
        ) -> dict:
            return {
                "split": split,
                "class": class_name,
                "label": int(label),
                "file_count": len(files),
                "event_count": event_count,
                "batch_size": int(batch_size),
                "sequence_type": spec.get("sequence_type"),
                "min_pt": spec.get("min_pt"),
                "eval_sequence_type": spec.get("eval_sequence_type"),
                "eval_min_pt": spec.get("eval_min_pt"),
                "sampling_weight": spec.get("weight"),
            }

        if self._multi_class:
            rows = []
            for class_name, label in self._class_to_label.items():
                train_spec = self._class_specs[class_name]
                test_spec = self._test_class_specs.get(class_name, train_spec)
                train_enabled = (
                    train_spec["max_train_events"] != 0 and train_spec["weight"] > 0
                )
                val_enabled = train_spec["max_val_events"] != 0
                test_enabled = test_spec.get("max_test_events") != 0
                rows.append(
                    split_row(
                        "train",
                        class_name,
                        label,
                        self._train_files_per_class[class_name] if train_enabled else [],
                        train_spec,
                        train_spec["max_train_events"] if train_enabled else 0,
                        self.batch_size_train,
                    )
                )
                rows.append(
                    split_row(
                        "val",
                        class_name,
                        label,
                        self._val_files_per_class[class_name] if val_enabled else [],
                        train_spec,
                        train_spec["max_val_events"] if val_enabled else 0,
                        self.batch_size_val,
                    )
                )
                rows.append(
                    split_row(
                        "test",
                        class_name,
                        label,
                        (
                            self._test_files_per_class.get(class_name, [])
                            if test_enabled
                            else []
                        ),
                        test_spec,
                        test_spec.get("max_test_events") if test_enabled else 0,
                        self.batch_size_test,
                    )
                )
            return {
                "mode": "multi_class",
                "train_fraction": float(self.hparams.train_fraction),
                "split_seed": int(self.hparams.split_seed),
                "class_to_label": dict(self._class_to_label),
                "rows": rows,
            }

        rows = [
            {
                "split": "train",
                "class": "all",
                "label": int(self.hparams.jet_type_label),
                "file_count": len(self.parquet_files_train),
                "event_count": None,
                "batch_size": int(self.batch_size_train),
                "sequence_type": self.hparams.sequence_type,
                "min_pt": self.hparams.min_pt,
                "eval_sequence_type": None,
                "eval_min_pt": None,
                "sampling_weight": 1.0,
            },
            {
                "split": "val",
                "class": "all",
                "label": int(self.hparams.jet_type_label),
                "file_count": len(self.parquet_files_val),
                "event_count": None,
                "batch_size": int(self.batch_size_val),
                "sequence_type": self.hparams.sequence_type,
                "min_pt": self.hparams.min_pt,
                "eval_sequence_type": None,
                "eval_min_pt": None,
                "sampling_weight": 1.0,
            },
            {
                "split": "test",
                "class": "all",
                "label": int(self.hparams.jet_type_label),
                "file_count": len(_dataset_files(self.parquet_files_test or [])),
                "event_count": None,
                "batch_size": int(self.batch_size_test),
                "sequence_type": self.hparams.sequence_type,
                "min_pt": self.hparams.min_pt,
                "eval_sequence_type": None,
                "eval_min_pt": None,
                "sampling_weight": 1.0,
            },
        ]
        return {
            "mode": "single_class",
            "train_fraction": float(self.hparams.train_fraction),
            "split_seed": int(self.hparams.split_seed),
            "class_to_label": {"all": int(self.hparams.jet_type_label)},
            "rows": rows,
        }

    @staticmethod
    def _normalize_class_specs(
        raw_specs,
        sequence_type: str,
        min_pt: Optional[float],
        class_sequence_types: Optional[dict],
        class_min_pts: Optional[dict],
        class_eval_sequence_types: Optional[dict],
        class_eval_min_pts: Optional[dict],
        class_sampling_weights: Optional[dict],
        max_files_per_class: Optional[dict],
        max_train_events_per_class: Optional[dict],
        max_val_events_per_class: Optional[dict],
        max_test_events_per_class: Optional[dict],
    ) -> dict[str, dict]:
        specs = {}
        class_sequence_types = class_sequence_types or {}
        class_min_pts = class_min_pts or {}
        class_eval_sequence_types = class_eval_sequence_types or {}
        class_eval_min_pts = class_eval_min_pts or {}
        class_sampling_weights = class_sampling_weights or {}
        max_files_per_class = max_files_per_class or {}
        max_train_events_per_class = max_train_events_per_class or {}
        max_val_events_per_class = max_val_events_per_class or {}
        max_test_events_per_class = max_test_events_per_class or {}

        for name, value in raw_specs.items():
            if isinstance(value, Mapping) and any(
                key in value for key in ("paths", "files", "manifest")
            ):
                paths = value.get("paths", value.get("files", value.get("manifest")))
                spec_sequence_type = value.get(
                    "sequence_type", class_sequence_types.get(name, sequence_type)
                )
                spec_min_pt = value.get("min_pt", class_min_pts.get(name, min_pt))
                eval_sequence_type = value.get(
                    "eval_sequence_type", class_eval_sequence_types.get(name)
                )
                eval_min_pt = value.get("eval_min_pt", class_eval_min_pts.get(name))
                spec_weight = value.get("weight", class_sampling_weights.get(name, 1.0))
                spec_max_files = value.get("max_files", max_files_per_class.get(name))
                max_train_events = value.get(
                    "max_train_events", max_train_events_per_class.get(name)
                )
                max_val_events = value.get("max_val_events", max_val_events_per_class.get(name))
                max_test_events = value.get(
                    "max_test_events", max_test_events_per_class.get(name)
                )
            else:
                paths = value
                spec_sequence_type = class_sequence_types.get(name, sequence_type)
                spec_min_pt = class_min_pts.get(name, min_pt)
                eval_sequence_type = class_eval_sequence_types.get(name)
                eval_min_pt = class_eval_min_pts.get(name)
                spec_weight = class_sampling_weights.get(name, 1.0)
                spec_max_files = max_files_per_class.get(name)
                max_train_events = max_train_events_per_class.get(name)
                max_val_events = max_val_events_per_class.get(name)
                max_test_events = max_test_events_per_class.get(name)

            if spec_sequence_type not in SEQUENCE_SCHEMAS:
                raise ValueError(
                    f"Unknown sequence_type {spec_sequence_type!r} for class {name!r}. "
                    f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
                )
            if eval_sequence_type is not None and eval_sequence_type not in SEQUENCE_SCHEMAS:
                raise ValueError(
                    f"Unknown eval_sequence_type {eval_sequence_type!r} for class {name!r}. "
                    f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
                )
            for split, max_events in (
                ("train", max_train_events),
                ("val", max_val_events),
                ("test", max_test_events),
            ):
                if max_events is not None and int(max_events) < 0:
                    raise ValueError(
                        f"max_{split}_events must be non-negative for class {name!r}"
                    )
            specs[name] = {
                "paths": paths,
                "sequence_type": spec_sequence_type,
                "min_pt": None if spec_min_pt is None else float(spec_min_pt),
                "eval_sequence_type": eval_sequence_type,
                "eval_min_pt": None if eval_min_pt is None else float(eval_min_pt),
                "weight": float(spec_weight),
                "max_files": None if spec_max_files is None else int(spec_max_files),
                "max_train_events": (
                    None if max_train_events is None else int(max_train_events)
                ),
                "max_val_events": None if max_val_events is None else int(max_val_events),
                "max_test_events": None if max_test_events is None else int(max_test_events),
            }
        return specs

    def _dataset(
        self,
        parquet_files,
        batch_size: int,
        shuffle_row_groups: bool,
        jet_type_label: Optional[int] = None,
        sequence_type: Optional[str] = None,
        min_pt: Optional[float] = None,
        event_filter_sequence_type: Optional[str] = None,
        event_filter_min_pt: Optional[float] = None,
        max_events: Optional[int] = None,
    ):
        return OrbitParquetDataset(
            parquet_files=parquet_files,
            sequence_type=sequence_type or self.hparams.sequence_type,
            max_sequence_length=self._max_sequence_length,
            batch_size=batch_size,
            shuffle_row_groups=shuffle_row_groups,
            shuffle_seed=self.hparams.shuffle_seed,
            mask_column=self.hparams.mask_column,
            mask_min_value=self.hparams.mask_min_value,
            jet_type_label=(
                jet_type_label if jet_type_label is not None else self.hparams.jet_type_label
            ),
            min_pt=min_pt if min_pt is not None else self.hparams.min_pt,
            event_filter_sequence_type=event_filter_sequence_type,
            event_filter_min_pt=event_filter_min_pt,
            max_events=max_events,
            pid_cfg=self.hparams.get("pid_cfg"),
            include_energy=self.hparams.include_energy,
            energy_shift=self.hparams.energy_shift,
            auxiliary_jet_sequence_type=self.hparams.auxiliary_jet_sequence_type,
            max_auxiliary_jets=self.hparams.max_auxiliary_jets,
        )

    def _loader(self, dataset, persistent_workers: bool = True):
        kwargs = {
            "batch_size": None,
            "num_workers": self.hparams.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": persistent_workers and self.hparams.num_workers > 0,
        }
        if self.hparams.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        if self._multi_class:
            datasets = {
                name: self._dataset(
                    self._train_files_per_class[name],
                    batch_size=self.batch_size_train,
                    shuffle_row_groups=self.hparams.shuffle_train,
                    jet_type_label=label,
                    sequence_type=self._class_specs[name]["sequence_type"],
                    min_pt=self._class_specs[name]["min_pt"],
                    max_events=self._class_specs[name]["max_train_events"],
                )
                for name, label in self._class_to_label.items()
                if self._class_specs[name]["max_train_events"] != 0
                and self._class_specs[name]["weight"] > 0
            }
            if not datasets:
                raise ValueError("At least one class must be enabled for training")
            return self._loader(
                WeightedClassDataset(
                    datasets,
                    weights={name: self._class_specs[name]["weight"] for name in datasets},
                    seed=self.hparams.shuffle_seed,
                )
            )
        return self._loader(
            self._dataset(
                self.parquet_files_train,
                batch_size=self.batch_size_train,
                shuffle_row_groups=self.hparams.shuffle_train,
            )
        )

    def val_dataloader(self):
        if self._multi_class:
            datasets = [
                self._dataset(
                    self._val_files_per_class[name],
                    batch_size=self.batch_size_val,
                    shuffle_row_groups=False,
                    jet_type_label=label,
                    sequence_type=self._class_specs[name]["sequence_type"],
                    min_pt=self._class_specs[name]["min_pt"],
                    event_filter_sequence_type=self._class_specs[name]["eval_sequence_type"],
                    event_filter_min_pt=self._class_specs[name]["eval_min_pt"],
                    max_events=self._class_specs[name]["max_val_events"],
                )
                for name, label in self._class_to_label.items()
                if self._class_specs[name]["max_val_events"] != 0
            ]
            if not datasets:
                raise ValueError("At least one class must be enabled for validation")
            return self._loader(ChainDataset(datasets))
        return self._loader(
            self._dataset(
                self.parquet_files_val,
                batch_size=self.batch_size_val,
                shuffle_row_groups=False,
            )
        )

    def test_dataloader(self):
        if self._multi_class and self._test_files_per_class:
            datasets = [
                self._dataset(
                    self._test_files_per_class[name],
                    batch_size=self.batch_size_test,
                    shuffle_row_groups=False,
                    jet_type_label=label,
                    sequence_type=self._test_class_specs[name]["sequence_type"],
                    min_pt=self._test_class_specs[name]["min_pt"],
                    event_filter_sequence_type=self._test_class_specs[name][
                        "eval_sequence_type"
                    ],
                    event_filter_min_pt=self._test_class_specs[name]["eval_min_pt"],
                    max_events=self._test_class_specs[name]["max_test_events"],
                )
                for name, label in self._class_to_label.items()
                if self._test_class_specs[name]["max_test_events"] != 0
            ]
            if not datasets:
                raise ValueError("At least one class must be enabled for testing")
            return self._loader(ChainDataset(datasets), persistent_workers=False)
        if not self.parquet_files_test:
            raise ValueError("Provide parquet_files_test from a separate test directory.")
        return self._loader(
            self._dataset(
                self.parquet_files_test,
                batch_size=self.batch_size_test,
                shuffle_row_groups=False,
            ),
            persistent_workers=False,
        )


def __getattr__(name):
    # Backward compatibility: the canonical data module moved to
    # gabbro.data.orbit_canonical, which itself imports this module, so it is
    # resolved lazily. Saved run configs still reference this path.
    if name == "CanonicalOrbitParquetDataModule":
        from gabbro.data.orbit_canonical import CanonicalOrbitDataModule

        return CanonicalOrbitDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
