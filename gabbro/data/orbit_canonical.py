"""Group-balanced canonical ORBIT data module over parquet and/or ROOT sources.

Each process declares ``format: parquet`` (default; collide-2v manifests, split
into train/val by file) or ``format: root`` (e.g. one merged Phase-2 L1 ntuple
per process, split by entry ranges). Quotas, labels, class balancing and test
suites behave identically for both.
"""

from collections.abc import Mapping
from typing import Optional

import lightning as L
import torch
from torch.utils.data import ChainDataset, DataLoader

from gabbro.data.orbit_common import (
    SEQUENCE_SCHEMAS,
    OrbitPreprocessor,
    OrbitSequenceDataset,
    WeightedClassDataset,
    resolve_process_quotas,
)
from gabbro.data.orbit_parquet import (
    PARQUET_SUFFIXES,
    OrbitParquetDataset,
    _dataset_files,
    deterministic_file_split,
)
from gabbro.data.orbit_root import (
    ROOT_SUFFIXES,
    OrbitRootDataset,
    root_files,
    validate_entry_fraction,
)

SOURCE_FORMATS = ("parquet", "root")
DEFAULT_ROOT_OPTIONS = {
    "tree": "Events",
    "columns": None,
    "sort_by_pt": True,
    "step_size": 10_000,
    # Same proportions as the collide-2v canonical split: 20% test, and the
    # remaining 80% split 80/20 into train/val.
    "entry_splits": {"train": [0.0, 0.64], "val": [0.64, 0.8], "test": [0.8, 1.0]},
    # Entries are assigned to splits in blocks of this size, shuffled with the
    # data module's split_seed; null keeps contiguous entry ranges.
    "split_block_size": 1000,
}


def _looks_like(paths, suffixes: set[str]) -> bool:
    values = [paths] if isinstance(paths, str) else paths
    try:
        return any(str(value).lower().endswith(tuple(suffixes)) for value in values)
    except TypeError:
        return False


class CanonicalOrbitDataModule(L.LightningDataModule):
    """Group-balanced ORBIT loader with named, independently balanced test suites.

    This additive interface is used by the canonical experiments. The legacy
    :class:`~gabbro.data.orbit_parquet.OrbitParquetDataModule` remains
    unchanged for existing experiments.

    ``root`` holds the defaults for every ``format: root`` process (``tree``,
    ``columns``, ``sort_by_pt``, ``step_size``, ``entry_splits``); a process
    may override any of them with a key of the same name.
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
        auxiliary_jet_sequence_type: Optional[str] = None,
        max_auxiliary_jets: int = 5,
        root: Optional[Mapping] = None,
        pin_memory: Optional[bool] = None,
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

        unknown_root_options = set(dict(root or {})) - set(DEFAULT_ROOT_OPTIONS)
        if unknown_root_options:
            raise ValueError(f"Unknown root options {sorted(unknown_root_options)}")
        self._root_defaults = {**DEFAULT_ROOT_OPTIONS, **dict(root or {})}

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
            if spec["format"] == "root":
                # One file list for both splits; the entry ranges separate them.
                files = self._root_files(spec["paths"], process_name)
                train_files, val_files = files, files
            else:
                train_files, val_files = deterministic_file_split(
                    spec["paths"], train_fraction=train_fraction, split_seed=split_seed
                )
            self._train_files[process_name] = train_files
            self._val_files[process_name] = val_files
        self._test_files = {
            suite_name: {
                process_name: (
                    self._root_files(spec["paths"], process_name)
                    if spec["format"] == "root"
                    else _dataset_files(spec["paths"])
                )
                for process_name, spec in specs.items()
            }
            for suite_name, specs in self._test_suite_specs.items()
        }
        self._check_root_split_overlaps()

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
        self.selected_jet_features = (
            OrbitPreprocessor(
                SEQUENCE_SCHEMAS[auxiliary_jet_sequence_type]["prefix"]
            ).output_features
            if auxiliary_jet_sequence_type is not None
            else []
        )
        self.save_hyperparameters()
        self.hparams["selected_features"] = self.selected_features
        self.hparams["selected_jet_features"] = self.selected_jet_features
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

    def _normalize_process_specs(
        self, raw_specs: Mapping[str, Mapping], sequence_type: str, min_pt: Optional[float]
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
            source_format = str(raw_spec.get("format", "parquet"))
            if source_format not in SOURCE_FORMATS:
                raise ValueError(
                    f"Unknown format {source_format!r} for process {process_name!r}. "
                    f"Expected one of {SOURCE_FORMATS}"
                )
            spec = {
                "paths": paths,
                "group": str(raw_spec.get("group", process_name)),
                "sequence_type": process_sequence_type,
                "min_pt": raw_spec.get("min_pt", min_pt),
                "eval_sequence_type": eval_sequence_type,
                "eval_min_pt": raw_spec.get("eval_min_pt"),
            }
            if source_format == "root":
                spec["format"] = "root"
                spec["root"] = self._root_options(raw_spec, process_name)
            else:
                if _looks_like(paths, ROOT_SUFFIXES):
                    raise ValueError(
                        f"Process {process_name!r} points at ROOT files; set `format: root`"
                    )
                spec["format"] = "parquet"
            specs[str(process_name)] = spec
        return specs

    def _root_options(self, raw_spec: Mapping, process_name: str) -> dict:
        options = {
            key: raw_spec.get(key, default) for key, default in self._root_defaults.items()
        }
        if not options["columns"]:
            raise ValueError(
                f"ROOT process {process_name!r} needs a `columns` mapping "
                "(set root.columns in the data config)"
            )
        options["columns"] = {
            collection: dict(mapping) for collection, mapping in dict(options["columns"]).items()
        }
        splits = {
            str(split): validate_entry_fraction(
                fraction, f"process {process_name!r} entry_splits.{split}"
            )
            for split, fraction in dict(options["entry_splits"]).items()
        }
        ordered = sorted(splits.items(), key=lambda item: item[1])
        for (name_a, (_, stop_a)), (name_b, (start_b, _)) in zip(ordered, ordered[1:]):
            if start_b < stop_a:
                raise ValueError(
                    f"process {process_name!r}: entry_splits {name_a} and {name_b} overlap"
                )
        options["entry_splits"] = {split: list(fraction) for split, fraction in splits.items()}
        return options

    @staticmethod
    def _root_files(paths, process_name: str) -> list[str]:
        files = root_files(paths)
        if not files:
            raise ValueError(f"ROOT process {process_name!r} resolved to no files")
        if _looks_like(files, PARQUET_SUFFIXES):
            raise ValueError(
                f"ROOT process {process_name!r} points at parquet files; "
                "remove `format: root`"
            )
        return files

    def _check_root_split_overlaps(self) -> None:
        """Reject a ROOT file whose test range overlaps a train/val range elsewhere.

        Per-process ``entry_splits`` overrides make this possible when the same
        file appears in train_val_processes and in a test suite. Fractions only
        index the same shuffled blocks if every use shares ``split_block_size``
        (the seed is global), so that is required too.
        """
        used: dict[tuple[str, str], list[tuple[str, str, list[float]]]] = {}
        block_sizes: dict[tuple[str, str], tuple[Optional[int], str]] = {}
        uses = [
            (split, process_name, spec, self._train_files[process_name])
            for process_name, spec in self._train_specs.items()
            for split in ("train", "val")
        ] + [
            ("test", process_name, spec, self._test_files[suite_name][process_name])
            for suite_name, specs in self._test_suite_specs.items()
            for process_name, spec in specs.items()
        ]
        for split, process_name, spec, files in uses:
            if spec["format"] != "root":
                continue
            fraction = self._entry_fraction(spec, split, process_name)
            for file in files:
                key = (file, spec["root"]["tree"])
                block_size = spec["root"]["split_block_size"]
                other_block_size, other_process = block_sizes.setdefault(
                    key, (block_size, process_name)
                )
                if block_size != other_block_size:
                    raise ValueError(
                        f"{file}: split_block_size {block_size} for {process_name!r} differs "
                        f"from {other_block_size} for {other_process!r}"
                    )
                for other_split, other_process, other_fraction in used.get(key, []):
                    overlaps = fraction[0] < other_fraction[1] and other_fraction[0] < fraction[1]
                    if other_split != split and overlaps:
                        raise ValueError(
                            f"{file}: {split} entries of {process_name!r} {fraction} overlap "
                            f"{other_split} entries of {other_process!r} {other_fraction}"
                        )
                used.setdefault(key, []).append((split, process_name, fraction))

    @staticmethod
    def _entry_fraction(spec: Mapping, split: str, process_name: str) -> list[float]:
        splits = spec["root"]["entry_splits"]
        if split not in splits:
            raise ValueError(f"ROOT process {process_name!r} has no entry_splits.{split}")
        return splits[split]

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
        split: str,
        shuffle: bool,
        process_name: str,
        test_suite_name: Optional[str] = None,
        apply_eval_filter: bool = False,
    ) -> OrbitSequenceDataset:
        common = dict(
            sequence_type=spec["sequence_type"],
            max_sequence_length=self._max_sequence_length,
            batch_size=batch_size,
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
            auxiliary_jet_sequence_type=self.hparams.auxiliary_jet_sequence_type,
            max_auxiliary_jets=self.hparams.max_auxiliary_jets,
        )
        if spec["format"] == "root":
            options = spec["root"]
            return OrbitRootDataset(
                files,
                columns=options["columns"],
                tree_name=options["tree"],
                entry_fraction=self._entry_fraction(spec, split, process_name),
                split_block_size=options["split_block_size"],
                split_seed=self.hparams.split_seed,
                sort_by_pt=options["sort_by_pt"],
                step_size=options["step_size"],
                shuffle=shuffle,
                **common,
            )
        return OrbitParquetDataset(files, shuffle_row_groups=shuffle, **common)

    def _loader(self, dataset, *, persistent_workers: bool = True) -> DataLoader:
        pin_memory = self.hparams.get("pin_memory")
        kwargs = {
            "batch_size": None,
            "num_workers": self.hparams.num_workers,
            "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
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
                split="train",
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
                split="val",
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
                    split="test",
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

    def _summary_row(self, split, suite, process_name, spec, files, event_count) -> dict:
        row = {
            "split": split,
            "suite": suite,
            "group": spec["group"],
            "process": process_name,
            "label": self._group_to_label[spec["group"]],
            "file_count": len(files),
            "event_count": event_count,
        }
        if spec["format"] == "root":
            row["format"] = "root"
            row["entry_fraction"] = self._entry_fraction(spec, split, process_name)
            row["split_block_size"] = spec["root"]["split_block_size"]
        return row

    def data_split_summary(self) -> dict:
        rows = []
        for split, files_by_process, quotas in (
            ("train", self._train_files, self._train_quotas),
            ("val", self._val_files, self._val_quotas),
        ):
            for process_name, spec in self._train_specs.items():
                rows.append(
                    self._summary_row(
                        split,
                        None,
                        process_name,
                        spec,
                        files_by_process[process_name],
                        quotas[process_name],
                    )
                )
        for suite_name, specs in self._test_suite_specs.items():
            for process_name, spec in specs.items():
                rows.append(
                    self._summary_row(
                        "test",
                        suite_name,
                        process_name,
                        spec,
                        self._test_files[suite_name][process_name],
                        self._test_quotas[suite_name][process_name],
                    )
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


# Name used by existing configs, scripts and saved run configs.
CanonicalOrbitParquetDataModule = CanonicalOrbitDataModule
