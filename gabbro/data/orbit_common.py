"""Source-agnostic building blocks shared by the ORBIT parquet and ROOT loaders.

Everything here operates on awkward batches whose fields already use the
internal ORBIT column names (``L1T_PUPPIPart_PT`` ...). Each source module
(``orbit_parquet``, ``orbit_root``) only has to find its files and yield such
batches; feature selection, preprocessing, padding, PID mapping, event quotas
and class balancing are shared.
"""

import os
import sys
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Optional

import awkward as ak
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


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


def read_path_manifest(
    path: Path, data_suffixes: set[str], seen: set[Path] | None = None
) -> list[str]:
    """Read ORBIT-style text manifests containing one data path per line."""
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive path manifest detected: {path}")
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
            files.extend(expand_path_entry(value_path, data_suffixes, seen=seen))
    return files


def expand_path_entry(
    path, data_suffixes: set[str], seen: set[Path] | None = None
) -> list[str]:
    """Expand one entry: manifests (and any existing non-data file) are read recursively."""
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.suffix.lower() in MANIFEST_SUFFIXES or (
        path.is_file() and path.suffix.lower() not in data_suffixes
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Path manifest does not exist: {path}")
        return read_path_manifest(path, data_suffixes, seen=seen)
    return [str(path)]


def expand_paths(paths, data_suffixes: set[str]) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded = []
    for path in paths:
        expanded.extend(expand_path_entry(path, data_suffixes))
    return expanded


def _log_worker_exception(message: str, source_kind: str = "parquet") -> None:
    worker_info = get_worker_info()
    worker = "main" if worker_info is None else f"{worker_info.id}/{worker_info.num_workers}"
    print(
        f"[ORBIT_{source_kind.upper()}_WORKER_ERROR] pid={os.getpid()} worker={worker} {message}",
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


class OrbitSequenceDataset(IterableDataset):
    """Stream pre-batched, padded ORBIT events from any source.

    Subclasses implement :meth:`_iter_source_batches`, which yields
    ``(batch, source_file, source_rows)`` tuples where ``batch`` is an awkward
    record array with (at least) the internal columns listed in
    :attr:`features`. This class handles everything after that: per-particle
    and per-event selection, preprocessing, padding, PID classes, auxiliary
    jets, event quotas and worker partitioning of those quotas.
    """

    source_kind = "source"

    def __init__(
        self,
        sequence_type: str = "particle",
        max_sequence_length: Optional[int] = None,
        batch_size: int = 32,
        shuffle: bool = False,
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
        auxiliary_jet_sequence_type: Optional[str] = None,
        max_auxiliary_jets: int = 5,
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

        schema = SEQUENCE_SCHEMAS[sequence_type]
        self.preprocessor = OrbitPreprocessor(
            schema["prefix"],
            include_energy=include_energy,
            energy_shift=energy_shift,
        )
        self.features = list(self.preprocessor.input_features)
        if auxiliary_jet_sequence_type is not None:
            if auxiliary_jet_sequence_type not in SEQUENCE_SCHEMAS:
                raise ValueError(
                    f"Unknown auxiliary_jet_sequence_type "
                    f"{auxiliary_jet_sequence_type!r}. Expected one of "
                    f"{sorted(SEQUENCE_SCHEMAS)}"
                )
            if not auxiliary_jet_sequence_type.startswith("jet_"):
                raise ValueError("auxiliary_jet_sequence_type must select a jet schema")
            if max_auxiliary_jets < 1:
                raise ValueError("max_auxiliary_jets must be positive when jets are enabled")
            auxiliary_schema = SEQUENCE_SCHEMAS[auxiliary_jet_sequence_type]
            self.auxiliary_jet_preprocessor = OrbitPreprocessor(
                auxiliary_schema["prefix"],
                include_energy=False,
                energy_shift=energy_shift,
            )
            self.features.extend(self.auxiliary_jet_preprocessor.input_features)
        else:
            self.auxiliary_jet_preprocessor = None
        self.auxiliary_jet_sequence_type = auxiliary_jet_sequence_type
        self.max_auxiliary_jets = int(max_auxiliary_jets)
        self.auxiliary_jet_source_length = (
            max(
                self.max_auxiliary_jets,
                SEQUENCE_SCHEMAS[auxiliary_jet_sequence_type]["max_sequence_length"],
            )
            if auxiliary_jet_sequence_type is not None
            else 0
        )
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
        self.features = list(dict.fromkeys(self.features))

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
        self.shuffle = shuffle
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

    def _iter_source_batches(
        self, worker_info, rng: Optional[np.random.Generator]
    ) -> Iterator[tuple[ak.Array, str, np.ndarray]]:
        """Yield ``(batch, source_file, source_rows)`` for this worker's share of the source.

        ``rng`` is set only when shuffling; it is seeded per epoch and must be
        consumed in a deterministic order. Implementations should stay lazy so
        iteration can stop as soon as the event quota is reached.
        """
        raise NotImplementedError

    def _describe(self) -> str:
        return (
            f"sequence_type={self.sequence_type!r} columns={self.features!r} "
            f"batch_size={self.batch_size} "
            f"max_sequence_length={self.max_sequence_length}"
        )

    def __iter__(self):
        yielded_events = 0
        try:
            worker_info = get_worker_info()
            rng = None
            if self.shuffle:
                rng = np.random.default_rng(self.shuffle_seed + self._iteration)
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
            _log_worker_exception("failed during dataset iteration setup", self.source_kind)
            raise

        if iteration_max_events is not None and iteration_max_events <= 0:
            return
        source_batches = self._iter_source_batches(worker_info, rng)
        try:
            # Check the quota before pulling the next raw batch, so a lazy
            # source never opens files it does not need.
            for batch, source_file, source_rows in source_batches:
                try:
                    converted_batches = list(
                        self._convert_batch(
                            batch, source_file=source_file, source_rows=source_rows
                        )
                    )
                except Exception:
                    _log_worker_exception(
                        f"failed while converting {self.source_kind} batch "
                        f"file={source_file!r} {self._describe()}",
                        self.source_kind,
                    )
                    raise
                for converted in converted_batches:
                    if iteration_max_events is not None:
                        remaining = iteration_max_events - yielded_events
                        if remaining <= 0:
                            return
                        batch_events = converted["part_features"].shape[0]
                        if batch_events > remaining:
                            converted = {
                                key: value[:remaining] for key, value in converted.items()
                            }
                    yielded_events += converted["part_features"].shape[0]
                    yield converted
                if iteration_max_events is not None and yielded_events >= iteration_max_events:
                    return
        finally:
            source_batches.close()

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
        if self.auxiliary_jet_preprocessor is not None:
            batch = self.auxiliary_jet_preprocessor.forward(batch)

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
        if self.auxiliary_jet_preprocessor is not None:
            jet_preprocessor = self.auxiliary_jet_preprocessor
            jet_stacked = ak.concatenate(
                [
                    batch[field][:, :, np.newaxis]
                    for field in jet_preprocessor.output_features
                ],
                axis=-1,
            )[non_empty_events]
            jet_lengths = ak.num(jet_stacked, axis=1)
            padded_jets = ak.pad_none(
                jet_stacked,
                self.auxiliary_jet_source_length,
                axis=1,
                clip=True,
            )
            filled_jets = ak.fill_none(
                padded_jets,
                [0.0] * len(jet_preprocessor.output_features),
                axis=1,
            )
            jet_features = torch.from_numpy(
                ak.to_numpy(filled_jets).astype(np.float32, copy=False)
            )
            valid_jet_lengths = torch.from_numpy(
                np.minimum(
                    ak.to_numpy(jet_lengths), self.auxiliary_jet_source_length
                ).astype(np.int64, copy=False)
            )
            source_jet_mask = (
                torch.arange(self.auxiliary_jet_source_length).unsqueeze(0)
                < valid_jet_lengths.unsqueeze(1)
            )
            # Sort the small dense stored-jet collection in Torch. This avoids
            # an Awkward argsort incompatibility with nullable Arrow list
            # columns while retaining the true leading-pT jets even if an
            # input file does not already order its collection.
            pt_index = jet_preprocessor.output_features.index(
                jet_preprocessor.pt_column
            )
            sort_score = jet_features[..., pt_index].masked_fill(
                ~source_jet_mask, -torch.inf
            )
            jet_order = torch.argsort(sort_score, dim=1, descending=True)
            jet_order = jet_order[:, : self.max_auxiliary_jets]
            converted["jet_features"] = torch.gather(
                jet_features,
                1,
                jet_order.unsqueeze(-1).expand(-1, -1, jet_features.shape[-1]),
            )
            converted["jet_mask"] = torch.gather(source_jet_mask, 1, jet_order)
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

