"""Iterable ROOT (TTree) loader for ORBIT particle and jet sequences.

Reads NanoAOD-style flat trees (``<Collection>_<variable>`` branches, e.g. the
Phase-2 L1 ``Events`` trees) with uproot. The configured branches are renamed
to the internal ORBIT column names (``L1T_PUPPIPart_PT`` ...) so that the
shared conversion in :class:`~gabbro.data.orbit_common.OrbitSequenceDataset`,
the preprocessing, the model and the plotting code are source-agnostic.

Unlike the parquet pipeline, ROOT samples typically come as one large merged
file per process, so train/val/test splits are made inside each file: its
entries are cut into blocks, the blocks are shuffled with a fixed seed, and
each split takes a fraction of the shuffled blocks (the analogue of the
seeded file-level split used for collide-2v manifests).
"""

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import awkward as ak
import numpy as np
import uproot

from gabbro.data.orbit_common import (
    SEQUENCE_SCHEMAS,
    OrbitSequenceDataset,
    _log_worker_exception,
    expand_paths,
)

ROOT_SUFFIXES = {".root"}


def root_files(paths) -> list[str]:
    """Resolve ROOT files, directories (``*.root`` inside) and text manifests."""
    files = []
    for path in expand_paths(paths, ROOT_SUFFIXES):
        if Path(path).is_dir():
            files.extend(str(file) for file in Path(path).glob("*.root"))
        else:
            files.append(str(path))
    return sorted(set(files))


def resolve_column_map(columns: Mapping) -> dict[str, str]:
    """Flatten a per-collection column map into ``internal column -> branch``.

    Each key is an internal collection prefix from ``SEQUENCE_SCHEMAS``; its
    ``prefix`` names the ROOT collection and every other entry maps an
    internal variable to a branch variable. Without ``prefix`` the values are
    full branch names.

    >>> resolve_column_map({"L1T_PUPPIPart": {"prefix": "L1PuppiCands", "PT": "pt"}})
    {'L1T_PUPPIPart_PT': 'L1PuppiCands_pt'}
    >>> resolve_column_map({"L1T_PUPPIPart": {"PuppiW": "L1PuppiCands_puppiWeight"}})
    {'L1T_PUPPIPart_PuppiW': 'L1PuppiCands_puppiWeight'}
    """
    resolved = {}
    for internal_prefix, mapping in dict(columns or {}).items():
        mapping = dict(mapping)
        branch_prefix = mapping.pop("prefix", None)
        for variable, branch_variable in mapping.items():
            branch = (
                str(branch_variable)
                if branch_prefix is None
                else f"{branch_prefix}_{branch_variable}"
            )
            resolved[f"{internal_prefix}_{variable}"] = branch
    return resolved


def validate_entry_fraction(fraction: Sequence[float], context: str) -> tuple[float, float]:
    if len(fraction) != 2:
        raise ValueError(f"{context}: entry fraction must be [start, stop], got {fraction!r}")
    start, stop = float(fraction[0]), float(fraction[1])
    if not 0.0 <= start < stop <= 1.0:
        raise ValueError(
            f"{context}: entry fraction must satisfy 0 <= start < stop <= 1, got {fraction!r}"
        )
    return start, stop


def split_entry_ranges(
    num_entries: int,
    fraction: Sequence[float],
    block_size: Optional[int] = None,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Return the sorted, merged ``[start, stop)`` entry ranges of one split.

    The entries are cut into blocks of ``block_size``, the blocks are permuted
    with ``seed``, and the split takes the blocks at positions
    ``[round(n_blocks * start), round(n_blocks * stop))`` of that permutation.
    Splits whose fractions tile ``[0, 1]`` are therefore disjoint and complete
    for any seed. ``block_size=None`` keeps one contiguous range instead.

    >>> split_entry_ranges(294000, (0.64, 0.8))
    [(188160, 235200)]
    >>> ranges = split_entry_ranges(294000, (0.0, 0.64), block_size=1000)
    >>> sum(stop - start for start, stop in ranges)
    188000
    >>> split_entry_ranges(10, (0.0, 0.6), block_size=2, seed=4)
    [(0, 2), (4, 6), (8, 10)]
    >>> split_entry_ranges(10, (0.6, 1.0), block_size=2, seed=4)
    [(2, 4), (6, 8)]
    """
    start, stop = fraction
    if block_size is None:
        first, last = int(round(num_entries * start)), int(round(num_entries * stop))
        return [(first, last)] if last > first else []
    if block_size < 1:
        raise ValueError("split_block_size must be positive")
    n_blocks = -(-num_entries // block_size)
    order = np.random.default_rng(seed).permutation(n_blocks)
    blocks = np.sort(order[int(round(n_blocks * start)) : int(round(n_blocks * stop))])
    ranges: list[tuple[int, int]] = []
    for block in blocks:
        first = int(block) * block_size
        last = min(first + block_size, num_entries)
        if ranges and ranges[-1][1] == first:
            ranges[-1] = (ranges[-1][0], last)
        else:
            ranges.append((first, last))
    return ranges


class OrbitRootDataset(OrbitSequenceDataset):
    """Stream pre-batched, padded ORBIT events from entry chunks of ROOT trees.

    ``entry_fraction`` selects this split's share of each file's shuffled
    ``split_block_size``-entry blocks (see :func:`split_entry_ranges`); the
    selected blocks are read in file order, in chunks of up to ``step_size``.
    """

    source_kind = "root"

    def __init__(
        self,
        root_paths,
        *,
        columns: Mapping,
        tree_name: str = "Events",
        entry_fraction: Sequence[float] = (0.0, 1.0),
        split_block_size: Optional[int] = 1000,
        split_seed: int = 42,
        sort_by_pt: bool = True,
        step_size: int = 10_000,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(shuffle=shuffle, **kwargs)
        files = root_files(root_paths)
        if not files:
            raise ValueError("root_paths must contain at least one ROOT file")
        if step_size < 1:
            raise ValueError("step_size must be positive")
        column_map = resolve_column_map(columns)
        unmapped = [column for column in self.features if column not in column_map]
        if unmapped:
            raise ValueError(
                f"No ROOT branch configured for internal columns {unmapped}; "
                "add them to the `columns` mapping"
            )
        # Construction stays cheap: files are opened lazily by each iterator.
        self.root_files = files
        self.tree_name = str(tree_name)
        self.entry_fraction = validate_entry_fraction(entry_fraction, "OrbitRootDataset")
        if split_block_size is not None and split_block_size < 1:
            raise ValueError("split_block_size must be positive or None")
        self.split_block_size = None if split_block_size is None else int(split_block_size)
        self.split_seed = int(split_seed)
        self.sort_by_pt = bool(sort_by_pt)
        self.step_size = int(step_size)
        self.branch_for = {column: column_map[column] for column in self.features}
        sequence_prefix = f"{SEQUENCE_SCHEMAS[self.sequence_type]['prefix']}_"
        # Every per-particle column of the main sequence is permuted together.
        self.sort_columns = [
            column for column in self.features if column.startswith(sequence_prefix)
        ]

    def _to_internal(self, arrays: ak.Array) -> ak.Array:
        fields = {column: arrays[branch] for column, branch in self.branch_for.items()}
        if self.sort_by_pt:
            # One permutation for all per-particle columns, applied before the
            # mask and the max_sequence_length truncation in _convert_batch,
            # so truncation drops the softest particles (as for collide-2v,
            # whose parquet collections are stored pT-descending).
            order = ak.argsort(
                fields[self.preprocessor.pt_column], axis=1, ascending=False, stable=True
            )
            for column in self.sort_columns:
                fields[column] = fields[column][order]
        return ak.zip(fields, depth_limit=1)

    def _iter_source_batches(self, worker_info, rng):
        files = list(self.root_files)
        if rng is not None:
            rng.shuffle(files)
        branches = list(dict.fromkeys(self.branch_for.values()))

        chunk_position = 0
        for file_path in files:
            chunk_start = None
            try:
                with uproot.open(file_path) as root_file:
                    tree = root_file[self.tree_name]
                    missing = sorted(set(branches) - set(tree.keys()))
                    if missing:
                        raise KeyError(f"{file_path}:{self.tree_name} has no branches {missing}")
                    entry_ranges = split_entry_ranges(
                        tree.num_entries,
                        self.entry_fraction,
                        block_size=self.split_block_size,
                        seed=self.split_seed,
                    )
                    if not entry_ranges:
                        warnings.warn(
                            f"{file_path}: entry fraction {self.entry_fraction} selects no "
                            f"entries ({tree.num_entries} entries, "
                            f"split_block_size={self.split_block_size})",
                            stacklevel=2,
                        )
                    chunks = [
                        (chunk_start, min(chunk_start + self.step_size, stop))
                        for start, stop in entry_ranges
                        for chunk_start in range(start, stop, self.step_size)
                    ]
                    if rng is not None:
                        rng.shuffle(chunks)

                    for chunk_start, chunk_stop in chunks:
                        assigned_to_worker = (
                            worker_info is None
                            or chunk_position % worker_info.num_workers == worker_info.id
                        )
                        chunk_position += 1
                        if not assigned_to_worker:
                            continue
                        batch = self._to_internal(
                            tree.arrays(branches, entry_start=chunk_start, entry_stop=chunk_stop)
                        )
                        for offset in range(0, len(batch), self.batch_size):
                            sub_batch = batch[offset : offset + self.batch_size]
                            rows = np.arange(
                                chunk_start + offset,
                                chunk_start + offset + len(sub_batch),
                                dtype=np.int64,
                            )
                            yield sub_batch, file_path, rows
            except Exception:
                _log_worker_exception(
                    "failed while reading ROOT "
                    f"file={file_path!r} tree={self.tree_name!r} chunk_start={chunk_start} "
                    f"{self._describe()}",
                    self.source_kind,
                )
                raise
