"""Iterable parquet loaders for ORBIT particle and jet sequences."""

import os
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import awkward as ak
import lightning as L
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import ChainDataset, DataLoader, IterableDataset, get_worker_info


SEQUENCE_SCHEMAS = {
    "particle": {
        "prefix": "L1T_PUPPIPart",
        "mask_column": "L1T_PUPPIPart_PuppiW",
        "mask_min_value": 0.05,
        "max_sequence_length": 128,
    },
    "particle_full": {
        "prefix": "L1T_PUPPIPart",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 500,
    },
    "jet_ak4": {
        "prefix": "L1T_JetAK4",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 14,
    },
    "jet_ak8": {
        "prefix": "L1T_JetAK8",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 7,
    },
    "jet_puppi_ak4": {
        "prefix": "L1T_JetPuppiAK4",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 14,
    },
    "jet_puppi_ak8": {
        "prefix": "L1T_JetPuppiAK8",
        "mask_column": None,
        "mask_min_value": 0.0,
        "max_sequence_length": 7,
    },
}

PID_CLASS_NAMES = (
    "neutral_hadron",
    "photon",
    "negative_hadron",
    "positive_hadron",
    "electron",
    "positron",
    "muon",
    "antimuon",
)


def map_pdg_charge_to_pid_class(pdg_id, charge):
    """Map raw PDG IDs and charges to the eight ORBIT hardware PID classes."""
    pid_class = ak.where(charge < 0, 2, ak.where(charge > 0, 3, 0))
    pid_class = ak.where(abs(pdg_id) == 22, 1, pid_class)
    pid_class = ak.where(pdg_id == 11, 4, pid_class)
    pid_class = ak.where(pdg_id == -11, 5, pid_class)
    pid_class = ak.where(pdg_id == 13, 6, pid_class)
    pid_class = ak.where(pdg_id == -13, 7, pid_class)
    return ak.values_astype(pid_class, np.int64)


MANIFEST_SUFFIXES = {".txt", ".list", ".lst"}
PARQUET_SUFFIXES = {".parquet"}


def _read_path_manifest(path: Path, seen: set[Path] | None = None) -> list[str]:
    """Read ORBIT-style text manifests containing one parquet path per line."""
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive parquet path manifest detected: {path}")
    seen.add(path)

    files = []
    with path.open() as manifest:
        for line in manifest:
            value = line.strip()
            if not value or value.lstrip().startswith("#"):
                continue
            value = os.path.expandvars(os.path.expanduser(value))
            value_path = Path(value)
            if not value_path.is_absolute():
                value_path = path.parent / value_path
            files.extend(_expand_path_entry(value_path, seen=seen))
    return files


def _expand_path_entry(path, seen: set[Path] | None = None) -> list[str]:
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.suffix.lower() in MANIFEST_SUFFIXES or (
        path.is_file() and path.suffix.lower() not in PARQUET_SUFFIXES
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Parquet path manifest does not exist: {path}")
        return _read_path_manifest(path, seen=seen)
    return [str(path)]


def _as_paths(paths) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded = []
    for path in paths:
        expanded.extend(_expand_path_entry(path))
    return expanded


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


def _log_worker_exception(message: str) -> None:
    worker_info = get_worker_info()
    worker = "main" if worker_info is None else f"{worker_info.id}/{worker_info.num_workers}"
    print(
        f"[ORBIT_PARQUET_WORKER_ERROR] pid={os.getpid()} worker={worker} {message}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


class OrbitPreprocessor:
    """Apply absolute-coordinate preprocessing without centering on a jet axis."""

    def __init__(
        self,
        feature_prefix: str,
        epsilon: float = 1e-8,
        include_energy: bool = False,
        energy_shift: float = 2.5,
    ):
        self.eta_column = f"{feature_prefix}_Eta"
        self.phi_column = f"{feature_prefix}_Phi"
        self.pt_column = f"{feature_prefix}_PT"
        self.energy_column = f"{feature_prefix}_E"
        self.output_features = [
            self.eta_column,
            f"{self.phi_column}_cos",
            f"{self.phi_column}_sin",
            self.pt_column,
        ]
        self.include_energy = bool(include_energy)
        self.energy_shift = float(energy_shift)
        if self.include_energy:
            self.output_features.append(self.energy_column)
        self.epsilon = epsilon

    @property
    def input_features(self) -> list[str]:
        features = [self.eta_column, self.phi_column, self.pt_column]
        if self.include_energy:
            features.append(self.energy_column)
        return features

    def forward(self, array: ak.Array) -> ak.Array:
        array = ak.with_field(array, array[self.eta_column] / 3, self.eta_column)
        array = ak.with_field(array, np.cos(array[self.phi_column]), f"{self.phi_column}_cos")
        array = ak.with_field(array, np.sin(array[self.phi_column]), f"{self.phi_column}_sin")
        array = ak.with_field(
            array,
            np.log(array[self.pt_column] + self.epsilon) - 1.8,
            self.pt_column,
        )
        if self.include_energy:
            array = ak.with_field(
                array,
                np.log(array[self.energy_column] + self.epsilon) - self.energy_shift,
                self.energy_column,
            )
        return array


class OrbitParquetDataset(IterableDataset):
    """Stream pre-batched, padded ORBIT events from parquet row groups."""

    def __init__(
        self,
        parquet_files,
        sequence_type: str = "particle",
        max_sequence_length: Optional[int] = None,
        batch_size: int = 32,
        shuffle_row_groups: bool = False,
        shuffle_seed: int = 42,
        mask_column: Optional[str] = None,
        mask_min_value: Optional[float] = None,
        jet_type_label: int = 0,
        process_label: int = 0,
        test_suite_label: int = 0,
        min_pt: Optional[float] = None,
        event_filter_sequence_type: Optional[str] = None,
        event_filter_min_pt: Optional[float] = None,
        max_events: Optional[int] = None,
        partition_max_events_across_workers: bool = False,
        return_raw_features: bool = False,
        return_event_metadata: bool = False,
        pid_cfg: Optional[Mapping] = None,
        include_energy: bool = False,
        energy_shift: float = 2.5,
    ):
        super().__init__()
        if sequence_type not in SEQUENCE_SCHEMAS:
            raise ValueError(
                f"Unknown sequence_type {sequence_type!r}. "
                f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
            )
        if (
            event_filter_sequence_type is not None
            and event_filter_sequence_type not in SEQUENCE_SCHEMAS
        ):
            raise ValueError(
                f"Unknown event_filter_sequence_type {event_filter_sequence_type!r}. "
                f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
            )
        parquet_files = _dataset_files(parquet_files)
        if not parquet_files:
            raise ValueError("parquet_files must contain at least one parquet file or directory")

        schema = SEQUENCE_SCHEMAS[sequence_type]
        self.preprocessor = OrbitPreprocessor(
            schema["prefix"],
            include_energy=include_energy,
            energy_shift=energy_shift,
        )
        self.features = list(self.preprocessor.input_features)
        pid_cfg = {} if pid_cfg is None else dict(pid_cfg)
        self.pid_enabled = bool(pid_cfg.get("enabled", False))
        self.pid_num_classes = int(pid_cfg.get("num_classes", len(PID_CLASS_NAMES)))
        if self.pid_enabled and not sequence_type.startswith("particle"):
            raise ValueError("PID features are only supported for particle sequence types")
        if self.pid_enabled and self.pid_num_classes != len(PID_CLASS_NAMES):
            raise ValueError(
                f"ORBIT PID mapping has {len(PID_CLASS_NAMES)} classes, "
                f"got num_classes={self.pid_num_classes}"
            )
        self.pid_column = f"{schema['prefix']}_PID" if self.pid_enabled else None
        self.charge_column = f"{schema['prefix']}_Charge" if self.pid_enabled else None
        if self.pid_enabled:
            self.features.extend([self.pid_column, self.charge_column])
        self.mask_column = schema["mask_column"] if mask_column is None else mask_column
        self.mask_min_value = (
            schema["mask_min_value"] if mask_min_value is None else mask_min_value
        )
        if self.mask_column is not None:
            self.features.append(self.mask_column)
        self.event_filter_pt_column = None
        if event_filter_sequence_type is not None:
            event_filter_prefix = SEQUENCE_SCHEMAS[event_filter_sequence_type]["prefix"]
            self.event_filter_pt_column = f"{event_filter_prefix}_PT"
            if self.event_filter_pt_column not in self.features:
                self.features.append(self.event_filter_pt_column)

        # Keep construction cheap.  In particular, do not open every parquet
        # file here to enumerate row groups: canonical datasets can contain
        # thousands of files while an event quota may consume only a handful.
        # Row groups are discovered lazily by each iterator and file scanning
        # stops as soon as max_events is reached.
        self.parquet_files = parquet_files

        self.sequence_type = sequence_type
        self.output_features = self.preprocessor.output_features
        self.raw_output_features = [
            self.preprocessor.pt_column,
            self.preprocessor.eta_column,
            self.preprocessor.phi_column,
        ]
        self.max_sequence_length = (
            schema["max_sequence_length"]
            if max_sequence_length is None
            else max_sequence_length
        )
        self.batch_size = batch_size
        self.shuffle_row_groups = shuffle_row_groups
        self.shuffle_seed = shuffle_seed
        self.jet_type_label = jet_type_label
        self.process_label = process_label
        self.test_suite_label = test_suite_label
        self.min_pt = min_pt
        self.event_filter_min_pt = event_filter_min_pt
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be non-negative")
        self.max_events = max_events
        self.partition_max_events_across_workers = partition_max_events_across_workers
        self.return_raw_features = return_raw_features
        self.return_event_metadata = return_event_metadata
        self._iteration = 0

    def __iter__(self):
        yielded_events = 0
        try:
            worker_info = get_worker_info()
            parquet_files = list(self.parquet_files)
            if self.shuffle_row_groups:
                rng = np.random.default_rng(self.shuffle_seed + self._iteration)
                rng.shuffle(parquet_files)
                self._iteration += 1
            iteration_max_events = self.max_events
            if (
                iteration_max_events is not None
                and worker_info is not None
                and self.partition_max_events_across_workers
            ):
                base, remainder = divmod(iteration_max_events, worker_info.num_workers)
                iteration_max_events = base + int(worker_info.id < remainder)
        except Exception:
            _log_worker_exception("failed during dataset iteration setup")
            raise

        row_group_position = 0
        for file_path in parquet_files:
            if iteration_max_events is not None and yielded_events >= iteration_max_events:
                return
            batch_idx = None
            row_group_idx = None
            try:
                parquet_file = pq.ParquetFile(file_path)
                row_groups = []
                file_row_offset = 0
                for row_group_idx in range(parquet_file.num_row_groups):
                    row_groups.append((row_group_idx, file_row_offset))
                    file_row_offset += parquet_file.metadata.row_group(row_group_idx).num_rows
                if self.shuffle_row_groups:
                    rng.shuffle(row_groups)

                for row_group_idx, row_group_row_offset in row_groups:
                    if iteration_max_events is not None and yielded_events >= iteration_max_events:
                        return
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
                        for converted in self._convert_batch(
                            ak.from_arrow(batch),
                            source_file=file_path,
                            source_rows=batch_rows,
                        ):
                            if iteration_max_events is not None:
                                remaining = iteration_max_events - yielded_events
                                if remaining <= 0:
                                    return
                                batch_events = converted["part_features"].shape[0]
                                if batch_events > remaining:
                                    converted = {
                                        key: value[:remaining]
                                        for key, value in converted.items()
                                    }
                                    batch_events = remaining
                            yielded_events += converted["part_features"].shape[0]
                            yield converted
            except Exception:
                _log_worker_exception(
                    "failed while reading parquet "
                    f"file={file_path!r} row_group={row_group_idx} "
                    f"sequence_type={self.sequence_type!r} columns={self.features!r} "
                    f"batch_size={self.batch_size} "
                    f"max_sequence_length={self.max_sequence_length} "
                    f"last_batch_idx={batch_idx}"
                )
                raise

    def _convert_batch(
        self,
        batch: ak.Array,
        source_file: Optional[str] = None,
        source_rows: Optional[np.ndarray] = None,
    ):
        event_selector = ak.ones_like(ak.num(batch[self.preprocessor.pt_column]), dtype=bool)
        if self.event_filter_pt_column is not None:
            filter_pts = batch[self.event_filter_pt_column]
            if self.event_filter_min_pt is None:
                event_selector = ak.num(filter_pts) > 0
            else:
                event_selector = ak.any(filter_pts >= self.event_filter_min_pt, axis=1)

        sequence_mask = ak.ones_like(batch[self.preprocessor.pt_column], dtype=bool)
        if self.min_pt is not None:
            sequence_mask = sequence_mask & (batch[self.preprocessor.pt_column] >= self.min_pt)
        if self.mask_column is not None:
            sequence_mask = sequence_mask & (batch[self.mask_column] > self.mask_min_value)

        raw_stacked = None
        if self.return_raw_features:
            raw_stacked = ak.concatenate(
                [
                    batch[field][sequence_mask][:, :, np.newaxis]
                    for field in self.raw_output_features
                ],
                axis=-1,
            )

        batch = self.preprocessor.forward(batch)

        stacked = ak.concatenate(
            [batch[field][sequence_mask][:, :, np.newaxis] for field in self.output_features],
            axis=-1,
        )
        event_lengths = ak.num(stacked, axis=1)
        non_empty_events = (event_lengths > 0) & event_selector
        if not ak.any(non_empty_events):
            return

        stacked = stacked[non_empty_events]
        pid_classes = None
        if self.pid_enabled:
            pid_classes = map_pdg_charge_to_pid_class(
                batch[self.pid_column],
                batch[self.charge_column],
            )[sequence_mask][non_empty_events]
        if raw_stacked is not None:
            raw_stacked = raw_stacked[non_empty_events][:, : self.max_sequence_length]
        event_lengths = event_lengths[non_empty_events]
        padded = ak.pad_none(stacked, self.max_sequence_length, axis=1, clip=True)
        filled = ak.fill_none(padded, [0.0] * len(self.output_features), axis=1)
        part_features = torch.from_numpy(ak.to_numpy(filled).astype(np.float32, copy=False))

        lengths = torch.from_numpy(
            np.minimum(ak.to_numpy(event_lengths), self.max_sequence_length).astype(
                np.int64, copy=False
            )
        )
        part_mask = torch.arange(self.max_sequence_length).unsqueeze(0) < lengths.unsqueeze(1)
        jet_type_labels = torch.full(
            (part_features.shape[0],),
            self.jet_type_label,
            dtype=torch.long,
        )

        converted = {
            "part_features": part_features,
            "part_mask": part_mask,
            "jet_type_labels": jet_type_labels,
            "process_labels": torch.full(
                (part_features.shape[0],), self.process_label, dtype=torch.long
            ),
            "test_suite_labels": torch.full(
                (part_features.shape[0],), self.test_suite_label, dtype=torch.long
            ),
        }
        if pid_classes is not None:
            padded_pid = ak.pad_none(pid_classes, self.max_sequence_length, axis=1, clip=True)
            filled_pid = ak.fill_none(padded_pid, -1, axis=1)
            converted["part_pid"] = torch.from_numpy(
                ak.to_numpy(filled_pid).astype(np.int64, copy=False)
            )
        if raw_stacked is not None:
            converted["raw_part_features"] = raw_stacked
        if self.return_event_metadata:
            if source_file is None or source_rows is None:
                raise ValueError("Event metadata requested without source provenance")
            selected_rows = np.asarray(source_rows)[ak.to_numpy(non_empty_events)]
            converted["source_files"] = np.full(
                part_features.shape[0], source_file, dtype=object
            )
            converted["source_rows"] = torch.from_numpy(
                selected_rows.astype(np.int64, copy=False)
            )
        yield converted


class WeightedClassDataset(IterableDataset):
    """Interleave pre-batched class datasets with configurable class weights."""

    def __init__(
        self,
        datasets: Mapping[str, IterableDataset],
        weights: Mapping[str, float],
        seed: int,
    ):
        super().__init__()
        self.datasets = dict(datasets)
        self.weights = {name: float(weights[name]) for name in self.datasets}
        self.seed = seed
        self._iteration = 0

    def __iter__(self):
        iterators = {name: iter(dataset) for name, dataset in self.datasets.items()}
        active = list(iterators)
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1

        while active:
            weights = np.asarray([self.weights[name] for name in active], dtype=np.float64)
            if np.any(weights < 0) or np.sum(weights) <= 0:
                raise ValueError("class_sampling_weights must be non-negative with positive sum")
            probabilities = weights / np.sum(weights)
            name = str(rng.choice(active, p=probabilities))
            try:
                yield next(iterators[name])
            except StopIteration:
                active.remove(name)


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


def balanced_group_process_quotas(
    process_specs: Mapping[str, Mapping], total_events: int
) -> dict[str, int]:
    """Split an event budget evenly by group and then by process.

    Remainders are assigned deterministically in mapping/declaration order.
    """
    if total_events < 1:
        raise ValueError("A balanced event budget must be positive")
    groups: dict[str, list[str]] = {}
    for process_name, spec in process_specs.items():
        group_name = str(spec.get("group", process_name))
        groups.setdefault(group_name, []).append(str(process_name))
    if not groups:
        raise ValueError("At least one process is required for balanced sampling")

    quotas: dict[str, int] = {}
    group_base, group_remainder = divmod(int(total_events), len(groups))
    for group_index, process_names in enumerate(groups.values()):
        group_budget = group_base + int(group_index < group_remainder)
        process_base, process_remainder = divmod(group_budget, len(process_names))
        for process_index, process_name in enumerate(process_names):
            quotas[process_name] = process_base + int(process_index < process_remainder)
    return quotas


def resolve_process_quotas(
    process_specs: Mapping[str, Mapping],
    *,
    event_budget: int,
    explicit_quotas: Optional[Mapping[str, int]] = None,
    context: str,
) -> dict[str, int]:
    """Return validated explicit quotas or the canonical balanced allocation."""
    if explicit_quotas is None:
        return balanced_group_process_quotas(process_specs, int(event_budget))

    quotas = {str(name): int(value) for name, value in explicit_quotas.items()}
    expected = set(map(str, process_specs))
    provided = set(quotas)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            f"{context} explicit event quotas do not match processes: "
            + ", ".join(details)
        )
    non_positive = {name: value for name, value in quotas.items() if value < 1}
    if non_positive:
        raise ValueError(f"{context} explicit event quotas must be positive: {non_positive}")
    if sum(quotas.values()) != int(event_budget):
        raise ValueError(
            f"{context} explicit event quotas sum to {sum(quotas.values())}, "
            f"not event_budget={int(event_budget)}"
        )
    return quotas


class CanonicalOrbitParquetDataModule(L.LightningDataModule):
    """Group-balanced ORBIT loader with named, independently balanced test suites.

    This additive interface is used by the canonical experiments. The legacy
    :class:`OrbitParquetDataModule` remains unchanged for existing experiments.
    """

    def __init__(
        self,
        train_val_processes: Mapping[str, Mapping],
        test_suites: Mapping[str, Mapping],
        train_event_budget: int = 200_000,
        val_event_budget: int = 200_000,
        train_event_quotas: Optional[Mapping[str, int]] = None,
        val_event_quotas: Optional[Mapping[str, int]] = None,
        sequence_type: str = "particle",
        max_sequence_length: Optional[int] = None,
        batch_size: int | Mapping[str, int] = 32,
        num_workers: int = 0,
        train_fraction: float = 0.8,
        split_seed: int = 42,
        shuffle_train: bool = True,
        shuffle_seed: int = 42,
        mask_column: Optional[str] = None,
        mask_min_value: Optional[float] = None,
        min_pt: Optional[float] = None,
        pid_cfg: Optional[dict] = None,
        return_raw_features: bool = False,
        return_event_metadata: bool = False,
        include_energy: bool = False,
        energy_shift: float = 2.5,
        **kwargs,
    ):
        super().__init__()
        if sequence_type not in SEQUENCE_SCHEMAS:
            raise ValueError(
                f"Unknown sequence_type {sequence_type!r}. "
                f"Expected one of {sorted(SEQUENCE_SCHEMAS)}"
            )
        if isinstance(batch_size, int):
            self.batch_size_train = self.batch_size_val = self.batch_size_test = batch_size
        else:
            required = {"train", "val", "test"}
            if not required.issubset(batch_size):
                raise ValueError("batch_size mapping must contain train, val, and test")
            self.batch_size_train = int(batch_size["train"])
            self.batch_size_val = int(batch_size["val"])
            self.batch_size_test = int(batch_size["test"])

        self._train_specs = self._normalize_process_specs(
            train_val_processes, sequence_type=sequence_type, min_pt=min_pt
        )
        if not self._train_specs:
            raise ValueError("train_val_processes must contain at least one process")
        self._test_suite_specs: dict[str, dict[str, dict]] = {}
        self._test_suite_budgets: dict[str, int] = {}
        self._test_suite_explicit_quotas: dict[str, Optional[Mapping[str, int]]] = {}
        for suite_name, suite_value in dict(test_suites or {}).items():
            if not isinstance(suite_value, Mapping) or "processes" not in suite_value:
                raise ValueError(
                    f"Test suite {suite_name!r} must contain a processes mapping"
                )
            budget = int(suite_value.get("event_budget", 20_000))
            self._test_suite_budgets[str(suite_name)] = budget
            self._test_suite_explicit_quotas[str(suite_name)] = suite_value.get(
                "event_quotas"
            )
            self._test_suite_specs[str(suite_name)] = self._normalize_process_specs(
                suite_value["processes"], sequence_type=sequence_type, min_pt=min_pt
            )
        if not self._test_suite_specs:
            raise ValueError("test_suites must contain at least one named suite")

        all_specs = list(self._train_specs.values()) + [
            spec for suite in self._test_suite_specs.values() for spec in suite.values()
        ]
        group_names = list(dict.fromkeys(spec["group"] for spec in all_specs))
        process_names = list(
            dict.fromkeys(
                list(self._train_specs)
                + [name for suite in self._test_suite_specs.values() for name in suite]
            )
        )
        self._group_to_label = {name: index for index, name in enumerate(group_names)}
        self._process_to_label = {name: index for index, name in enumerate(process_names)}
        self._test_suite_to_label = {
            name: index for index, name in enumerate(self._test_suite_specs)
        }

        self._train_quotas = resolve_process_quotas(
            self._train_specs,
            event_budget=int(train_event_budget),
            explicit_quotas=train_event_quotas,
            context="train",
        )
        self._val_quotas = resolve_process_quotas(
            self._train_specs,
            event_budget=int(val_event_budget),
            explicit_quotas=val_event_quotas,
            context="validation",
        )
        self._test_quotas = {
            suite_name: resolve_process_quotas(
                specs,
                event_budget=self._test_suite_budgets[suite_name],
                explicit_quotas=self._test_suite_explicit_quotas[suite_name],
                context=f"test suite {suite_name!r}",
            )
            for suite_name, specs in self._test_suite_specs.items()
        }

        self._train_files: dict[str, list[str]] = {}
        self._val_files: dict[str, list[str]] = {}
        for process_name, spec in self._train_specs.items():
            train_files, val_files = deterministic_file_split(
                spec["paths"], train_fraction=train_fraction, split_seed=split_seed
            )
            self._train_files[process_name] = train_files
            self._val_files[process_name] = val_files
        self._test_files = {
            suite_name: {
                process_name: _dataset_files(spec["paths"])
                for process_name, spec in specs.items()
            }
            for suite_name, specs in self._test_suite_specs.items()
        }

        self._max_sequence_length = (
            SEQUENCE_SCHEMAS[sequence_type]["max_sequence_length"]
            if max_sequence_length is None
            else int(max_sequence_length)
        )
        self.selected_features = OrbitPreprocessor(
            SEQUENCE_SCHEMAS[sequence_type]["prefix"],
            include_energy=include_energy,
            energy_shift=energy_shift,
        ).output_features
        self.save_hyperparameters()
        self.hparams["selected_features"] = self.selected_features
        self.hparams["max_sequence_length"] = self._max_sequence_length
        # Backward-compatible name consumed by existing callbacks.
        self.hparams["class_to_label"] = dict(self._group_to_label)
        self.hparams["group_to_label"] = dict(self._group_to_label)
        self.hparams["process_to_label"] = dict(self._process_to_label)
        self.hparams["process_to_group"] = {
            name: spec["group"] for name, spec in self._train_specs.items()
        }
        self.hparams["test_suite_to_label"] = dict(self._test_suite_to_label)
        self.hparams["class_specs"] = {
            group: next(spec for spec in all_specs if spec["group"] == group)
            for group in group_names
        }

    @staticmethod
    def _normalize_process_specs(
        raw_specs: Mapping[str, Mapping], sequence_type: str, min_pt: Optional[float]
    ) -> dict[str, dict]:
        specs = {}
        for process_name, raw_spec in dict(raw_specs or {}).items():
            if not isinstance(raw_spec, Mapping):
                raw_spec = {"paths": raw_spec}
            paths = raw_spec.get("paths", raw_spec.get("files", raw_spec.get("manifest")))
            if paths is None:
                raise ValueError(f"Process {process_name!r} is missing paths")
            process_sequence_type = str(raw_spec.get("sequence_type", sequence_type))
            if process_sequence_type not in SEQUENCE_SCHEMAS:
                raise ValueError(
                    f"Unknown sequence_type {process_sequence_type!r} "
                    f"for process {process_name!r}"
                )
            eval_sequence_type = raw_spec.get("eval_sequence_type")
            if eval_sequence_type is not None and eval_sequence_type not in SEQUENCE_SCHEMAS:
                raise ValueError(
                    f"Unknown eval_sequence_type {eval_sequence_type!r} "
                    f"for process {process_name!r}"
                )
            specs[str(process_name)] = {
                "paths": paths,
                "group": str(raw_spec.get("group", process_name)),
                "sequence_type": process_sequence_type,
                "min_pt": raw_spec.get("min_pt", min_pt),
                "eval_sequence_type": eval_sequence_type,
                "eval_min_pt": raw_spec.get("eval_min_pt"),
            }
        return specs

    @property
    def test_suite_names(self) -> list[str]:
        return list(self._test_suite_specs)

    def _dataset(
        self,
        files,
        spec: Mapping,
        batch_size: int,
        max_events: int,
        *,
        shuffle: bool,
        process_name: str,
        test_suite_name: Optional[str] = None,
        apply_eval_filter: bool = False,
    ) -> OrbitParquetDataset:
        return OrbitParquetDataset(
            parquet_files=files,
            sequence_type=spec["sequence_type"],
            max_sequence_length=self._max_sequence_length,
            batch_size=batch_size,
            shuffle_row_groups=shuffle,
            shuffle_seed=self.hparams.shuffle_seed,
            mask_column=self.hparams.mask_column,
            mask_min_value=self.hparams.mask_min_value,
            jet_type_label=self._group_to_label[spec["group"]],
            process_label=self._process_to_label[process_name],
            test_suite_label=(
                0
                if test_suite_name is None
                else self._test_suite_to_label[test_suite_name]
            ),
            min_pt=spec["min_pt"],
            event_filter_sequence_type=(
                spec["eval_sequence_type"] if apply_eval_filter else None
            ),
            event_filter_min_pt=spec["eval_min_pt"] if apply_eval_filter else None,
            max_events=max_events,
            partition_max_events_across_workers=True,
            return_raw_features=self.hparams.return_raw_features,
            return_event_metadata=self.hparams.return_event_metadata,
            pid_cfg=self.hparams.get("pid_cfg"),
            include_energy=self.hparams.include_energy,
            energy_shift=self.hparams.energy_shift,
        )

    def _loader(self, dataset, *, persistent_workers: bool = True) -> DataLoader:
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
        datasets = {
            process_name: self._dataset(
                self._train_files[process_name],
                spec,
                self.batch_size_train,
                self._train_quotas[process_name],
                shuffle=self.hparams.shuffle_train,
                process_name=process_name,
            )
            for process_name, spec in self._train_specs.items()
        }
        group_sizes: dict[str, int] = {}
        for spec in self._train_specs.values():
            group_sizes[spec["group"]] = group_sizes.get(spec["group"], 0) + 1
        weights = {
            process_name: 1.0 / group_sizes[spec["group"]]
            for process_name, spec in self._train_specs.items()
        }
        return self._loader(
            WeightedClassDataset(datasets, weights=weights, seed=self.hparams.shuffle_seed)
        )

    def val_dataloader(self):
        datasets = [
            self._dataset(
                self._val_files[process_name],
                spec,
                self.batch_size_val,
                self._val_quotas[process_name],
                shuffle=False,
                process_name=process_name,
                apply_eval_filter=True,
            )
            for process_name, spec in self._train_specs.items()
        ]
        return self._loader(ChainDataset(datasets))

    def test_dataloader(self):
        loaders = []
        for suite_name, specs in self._test_suite_specs.items():
            datasets = [
                self._dataset(
                    self._test_files[suite_name][process_name],
                    spec,
                    self.batch_size_test,
                    self._test_quotas[suite_name][process_name],
                    shuffle=False,
                    process_name=process_name,
                    test_suite_name=suite_name,
                    apply_eval_filter=True,
                )
                for process_name, spec in specs.items()
            ]
            loaders.append(
                self._loader(ChainDataset(datasets), persistent_workers=False)
            )
        return loaders

    def data_split_summary(self) -> dict:
        rows = []
        for split, files_by_process, quotas in (
            ("train", self._train_files, self._train_quotas),
            ("val", self._val_files, self._val_quotas),
        ):
            for process_name, spec in self._train_specs.items():
                rows.append(
                    {
                        "split": split,
                        "suite": None,
                        "group": spec["group"],
                        "process": process_name,
                        "label": self._group_to_label[spec["group"]],
                        "file_count": len(files_by_process[process_name]),
                        "event_count": quotas[process_name],
                    }
                )
        for suite_name, specs in self._test_suite_specs.items():
            for process_name, spec in specs.items():
                rows.append(
                    {
                        "split": "test",
                        "suite": suite_name,
                        "group": spec["group"],
                        "process": process_name,
                        "label": self._group_to_label[spec["group"]],
                        "file_count": len(self._test_files[suite_name][process_name]),
                        "event_count": self._test_quotas[suite_name][process_name],
                    }
                )
        return {
            "mode": "canonical_group_balanced",
            "train_fraction": float(self.hparams.train_fraction),
            "split_seed": int(self.hparams.split_seed),
            "group_to_label": dict(self._group_to_label),
            "process_to_label": dict(self._process_to_label),
            "test_suite_to_label": dict(self._test_suite_to_label),
            "rows": rows,
        }
