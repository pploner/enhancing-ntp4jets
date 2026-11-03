"""Tests for the ROOT (TTree) ORBIT loader and mixed-format canonical data module."""

import importlib.util
from collections import Counter
from pathlib import Path

import awkward as ak
import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import uproot
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from gabbro.data.orbit_canonical import CanonicalOrbitDataModule
from gabbro.data.orbit_parquet import OrbitParquetDataset
from gabbro.data.orbit_root import OrbitRootDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_test_root.py"
SPEC = importlib.util.spec_from_file_location("generate_test_root", SCRIPT_PATH)
generate_test_root = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_test_root)

COLUMNS = {
    "L1T_PUPPIPart": {
        "prefix": "L1PuppiCands",
        "PT": "pt",
        "Eta": "eta",
        "Phi": "phi",
        "PuppiW": "puppiWeight",
        "PID": "pdgId",
        "Charge": "charge",
    }
}
PID = {"enabled": True, "num_classes": 8}


def _random_events(n_events, seed=0, max_particles=40):
    """Per-event particle lists, stored pT-descending (the collide-2v convention)."""
    rng = np.random.default_rng(seed)
    pid_values = np.asarray([130, 22, -211, 211, 11, -11, 13, -13])
    charge_values = np.asarray([0, 0, -1, 1, -1, 1, -1, 1])
    events = []
    for _ in range(n_events):
        n = int(rng.integers(0, max_particles))
        index = rng.integers(0, len(pid_values), n)
        pt = (1.0 + rng.exponential(5.0, n)).astype(np.float32)
        order = np.argsort(-pt, kind="stable")
        events.append(
            {
                "pt": pt[order],
                "eta": rng.uniform(-4, 4, n).astype(np.float32)[order],
                "phi": rng.uniform(-np.pi, np.pi, n).astype(np.float32)[order],
                # Some weights fall below the 0.05 mask threshold.
                "puppiWeight": rng.uniform(0, 1, n).astype(np.float32)[order],
                "pdgId": pid_values[index][order],
                "charge": charge_values[index][order],
            }
        )
    return events


def _write_parquet(path, events):
    names = {
        "pt": "PT",
        "eta": "Eta",
        "phi": "Phi",
        "puppiWeight": "PuppiW",
        "pdgId": "PID",
        "charge": "Charge",
    }
    pq.write_table(
        pa.table(
            {
                f"L1T_PUPPIPart_{internal}": [event[variable].tolist() for event in events]
                for variable, internal in names.items()
            }
        ),
        path,
    )


def _write_root(path, events, shuffle_seed=1, tree="Events"):
    """Write NanoAOD-style branches with each event's particles in random order."""
    rng = np.random.default_rng(shuffle_seed)
    shuffled = [
        {k: v[perm] for k, v in e.items()}
        for e in events
        for perm in [rng.permutation(len(e["pt"]))]
    ]
    candidates = ak.zip(
        {
            variable: ak.Array([event[variable].astype(np.float32) for event in shuffled])
            for variable in events[0]
        }
    )
    with uproot.recreate(path) as root_file:
        root_file.mktree(tree, {"L1PuppiCands": candidates.type.content}).extend(
            {"L1PuppiCands": candidates}
        )


def _collect(dataset):
    return list(dataset)


def _all_rows(batches):
    return [int(row) for batch in batches for row in batch["source_rows"]]


def test_root_matches_parquet_for_identical_content(tmp_path):
    events = _random_events(60, seed=3)
    _write_parquet(tmp_path / "events.parquet", events)
    _write_root(tmp_path / "events.root", events)
    common = dict(
        batch_size=16,
        # Shorter than many events: only matches if ROOT is sorted before truncation.
        max_sequence_length=12,
        pid_cfg=PID,
        return_raw_features=True,
        return_event_metadata=True,
    )

    parquet_batches = _collect(OrbitParquetDataset(str(tmp_path / "events.parquet"), **common))
    root_batches = _collect(
        OrbitRootDataset(str(tmp_path / "events.root"), columns=COLUMNS, step_size=25, **common)
    )

    assert parquet_batches
    parquet = {
        k: torch.cat([b[k] for b in parquet_batches])
        for k in parquet_batches[0]
        if isinstance(parquet_batches[0][k], torch.Tensor)
    }
    root = {
        k: torch.cat([b[k] for b in root_batches])
        for k in root_batches[0]
        if isinstance(root_batches[0][k], torch.Tensor)
    }
    assert parquet.keys() == root.keys()
    for key in parquet:
        assert torch.equal(parquet[key], root[key]), key
    parquet_raw = [e for b in parquet_batches for e in b["raw_part_features"].to_list()]
    root_raw = [e for b in root_batches for e in b["raw_part_features"].to_list()]
    assert parquet_raw == root_raw


def test_without_sorting_root_keeps_stored_order(tmp_path):
    events = _random_events(20, seed=5)
    _write_root(tmp_path / "events.root", events)
    kwargs = dict(columns=COLUMNS, batch_size=50, max_sequence_length=64, pid_cfg=PID)

    unsorted = _collect(
        OrbitRootDataset(str(tmp_path / "events.root"), sort_by_pt=False, **kwargs)
    )
    sorted_ = _collect(OrbitRootDataset(str(tmp_path / "events.root"), sort_by_pt=True, **kwargs))

    def pts(batch):
        log_pt = batch["part_features"][..., 3]
        return [row[mask].tolist() for row, mask in zip(log_pt, batch["part_mask"])]

    sorted_pts = pts(sorted_[0])
    unsorted_pts = pts(unsorted[0])
    assert all(row == sorted(row, reverse=True) for row in sorted_pts)
    assert any(row != sorted(row, reverse=True) for row in unsorted_pts)
    assert [sorted(row) for row in unsorted_pts] == [sorted(row) for row in sorted_pts]


SPLITS = {"train": (0.0, 0.64), "val": (0.64, 0.8), "test": (0.8, 1.0)}


def _split_rows(path, block_size, seed=3):
    rows = {}
    for split, fraction in SPLITS.items():
        dataset = OrbitRootDataset(
            str(path),
            columns=COLUMNS,
            entry_fraction=fraction,
            split_block_size=block_size,
            split_seed=seed,
            step_size=7,
            batch_size=5,
            pid_cfg=PID,
            return_event_metadata=True,
        )
        rows[split] = _all_rows(_collect(dataset))
    return rows


def _non_empty_rows(events, rows):
    """Rows the loader keeps: at least one particle survives the PuppiW > 0.05 mask."""
    return {row for row in rows if np.any(events[row]["puppiWeight"] > 0.05)}


def test_shuffled_block_splits_are_disjoint_complete_and_seeded(tmp_path):
    events = _random_events(101, seed=7)
    _write_root(tmp_path / "events.root", events)

    rows = _split_rows(tmp_path / "events.root", block_size=8)
    all_rows = rows["train"] + rows["val"] + rows["test"]
    assert len(all_rows) == len(set(all_rows))
    assert set(all_rows) == _non_empty_rows(events, range(101))
    # 13 blocks of 8 (the last one has 5 entries) -> 8 / 2 / 3 blocks, and
    # every block belongs to exactly one split.
    blocks = {split: {row // 8 for row in split_rows} for split, split_rows in rows.items()}
    assert [len(blocks[split]) for split in SPLITS] == [8, 2, 3]
    assert sum(len(b) for b in blocks.values()) == len(set().union(*blocks.values())) == 13
    # Blocks are shuffled, not a contiguous prefix, and read in file order.
    assert set(rows["train"]) != _non_empty_rows(events, range(len(rows["train"])))
    assert rows["val"] == sorted(rows["val"])
    assert _split_rows(tmp_path / "events.root", block_size=8) == rows
    assert _split_rows(tmp_path / "events.root", block_size=8, seed=4)["train"] != rows["train"]


def test_contiguous_splits_without_block_size(tmp_path):
    events = _random_events(101, seed=7)
    _write_root(tmp_path / "events.root", events)

    rows = _split_rows(tmp_path / "events.root", block_size=None)
    assert set(rows["train"]) == _non_empty_rows(events, range(0, 65))
    assert set(rows["val"]) == _non_empty_rows(events, range(65, 81))
    assert set(rows["test"]) == _non_empty_rows(events, range(81, 101))


def test_workers_share_chunks_without_duplicates_and_respect_quota(tmp_path):
    events = _random_events(90, seed=11)
    _write_root(tmp_path / "events.root", events)
    kwargs = dict(
        columns=COLUMNS, step_size=10, batch_size=4, pid_cfg=PID, return_event_metadata=True
    )

    full = OrbitRootDataset(str(tmp_path / "events.root"), **kwargs)
    loader_rows = _all_rows(DataLoader(full, batch_size=None, num_workers=2))
    assert sorted(loader_rows) == sorted(_all_rows(_collect(full)))
    assert len(loader_rows) == len(set(loader_rows))

    capped = OrbitRootDataset(
        str(tmp_path / "events.root"),
        max_events=21,
        partition_max_events_across_workers=True,
        **kwargs,
    )
    capped_rows = _all_rows(DataLoader(capped, batch_size=None, num_workers=2))
    assert len(capped_rows) == 21
    assert len(set(capped_rows)) == 21


def test_shuffle_changes_chunk_order_per_epoch_but_not_content(tmp_path):
    events = _random_events(80, seed=13)
    _write_root(tmp_path / "events.root", events)
    dataset = OrbitRootDataset(
        str(tmp_path / "events.root"),
        columns=COLUMNS,
        step_size=8,
        batch_size=8,
        shuffle=True,
        shuffle_seed=0,
        pid_cfg=PID,
        return_event_metadata=True,
    )
    epoch_1 = _all_rows(_collect(dataset))
    epoch_2 = _all_rows(_collect(dataset))
    assert epoch_1 != epoch_2
    assert sorted(epoch_1) == sorted(epoch_2)


def test_unmapped_columns_and_missing_branches_fail_clearly(tmp_path):
    events = _random_events(5, seed=17)
    _write_root(tmp_path / "events.root", events)
    no_pid = {
        "L1T_PUPPIPart": {
            k: v for k, v in COLUMNS["L1T_PUPPIPart"].items() if k not in {"PID", "Charge"}
        }
    }
    with pytest.raises(ValueError, match="L1T_PUPPIPart_PID"):
        OrbitRootDataset(str(tmp_path / "events.root"), columns=no_pid, pid_cfg=PID)

    wrong_branch = {"L1T_PUPPIPart": {**COLUMNS["L1T_PUPPIPart"], "PuppiW": "puppiW"}}
    dataset = OrbitRootDataset(str(tmp_path / "events.root"), columns=wrong_branch, pid_cfg=PID)
    with pytest.raises(KeyError, match="L1PuppiCands_puppiW"):
        _collect(dataset)


def _canonical(tmp_path, *, tt_spec=None, test_spec=None, **kwargs):
    events = _random_events(100, seed=19)
    root_path = tmp_path / "tt.root"
    _write_root(root_path, events)
    parquet_paths = []
    for index in range(3):
        path = tmp_path / f"qcd_{index}.parquet"
        _write_parquet(path, _random_events(30, seed=100 + index))
        parquet_paths.append(str(path))
    tt_spec = tt_spec or {"format": "root", "paths": str(root_path), "group": "tt"}
    test_spec = test_spec or {"format": "root", "paths": str(root_path), "group": "tt"}
    return CanonicalOrbitDataModule(
        train_val_processes={
            "TT": tt_spec,
            "QCD": {"paths": parquet_paths, "group": "QCD"},
        },
        test_suites={"ttbar": {"event_budget": 15, "processes": {"TT": test_spec}}},
        train_event_budget=60,
        val_event_budget=20,
        train_event_quotas={"TT": 40, "QCD": 20},
        val_event_quotas={"TT": 10, "QCD": 10},
        batch_size=8,
        pid_cfg=PID,
        return_event_metadata=True,
        root={"columns": COLUMNS, "step_size": 16, "split_block_size": 5},
        **kwargs,
    )


def test_canonical_datamodule_mixes_root_and_parquet_processes(tmp_path):
    dm = _canonical(tmp_path)

    def label_counts(loader):
        counts = Counter()
        for batch in loader:
            counts.update(batch["process_labels"].tolist())
        return counts

    def tt_rows(loader):
        return [
            int(row)
            for batch in loader
            for row, label in zip(batch["source_rows"], batch["process_labels"])
            if int(label) == tt
        ]

    tt, qcd = dm.hparams.process_to_label["TT"], dm.hparams.process_to_label["QCD"]
    assert label_counts(dm.train_dataloader()) == {tt: 40, qcd: 20}
    assert label_counts(dm.val_dataloader()) == {tt: 10, qcd: 10}

    train_rows = tt_rows(dm.train_dataloader())
    val_rows = tt_rows(dm.val_dataloader())
    test_rows = [row for loader in dm.test_dataloader() for row in tt_rows(loader)]
    assert (len(train_rows), len(val_rows), len(test_rows)) == (40, 10, 15)
    assert not set(train_rows) & set(val_rows)
    assert not (set(train_rows) | set(val_rows)) & set(test_rows)

    rows = {(row["split"], row["process"]): row for row in dm.data_split_summary()["rows"]}
    assert rows[("train", "TT")]["entry_fraction"] == [0.0, 0.64]
    assert rows[("test", "TT")]["entry_fraction"] == [0.8, 1.0]
    assert rows[("test", "TT")]["split_block_size"] == 5
    assert "entry_fraction" not in rows[("train", "QCD")]
    assert rows[("train", "QCD")]["file_count"] + rows[("val", "QCD")]["file_count"] == 3


def test_canonical_rejects_root_split_leakage(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        _canonical(
            tmp_path,
            test_spec={
                "format": "root",
                "paths": str(tmp_path / "tt.root"),
                "group": "tt",
                "entry_splits": {"test": [0.5, 1.0]},
            },
        )
    with pytest.raises(ValueError, match="train and val overlap"):
        _canonical(
            tmp_path,
            tt_spec={
                "format": "root",
                "paths": str(tmp_path / "tt.root"),
                "group": "tt",
                "entry_splits": {"train": [0.0, 0.7], "val": [0.6, 0.8]},
            },
        )
    # Fractions of differently sized shuffled blocks select unrelated entries.
    with pytest.raises(ValueError, match="split_block_size"):
        _canonical(
            tmp_path,
            test_spec={
                "format": "root",
                "paths": str(tmp_path / "tt.root"),
                "group": "tt",
                "split_block_size": 10,
            },
        )


def test_canonical_requires_explicit_root_format(tmp_path):
    with pytest.raises(ValueError, match="set `format: root`"):
        _canonical(tmp_path, tt_spec={"paths": str(tmp_path / "tt.root"), "group": "tt"})


def test_old_canonical_import_path_still_resolves():
    from gabbro.data.orbit_parquet import CanonicalOrbitParquetDataModule

    assert CanonicalOrbitParquetDataModule is CanonicalOrbitDataModule
    located = hydra.utils.get_class("gabbro.data.orbit_parquet.CanonicalOrbitParquetDataModule")
    assert located is CanonicalOrbitDataModule


def test_generated_fixture_is_a_nanoaod_style_ttree(tmp_path):
    path = tmp_path / "smoke.root"
    generate_test_root.generate_root(path, n_events=50)
    tree = uproot.open(path)["Events"]
    assert tree.num_entries == 50
    assert {"nL1PuppiCands", "L1PuppiCands_pt", "L1PuppiCands_pdgId"} <= set(tree.keys())


def test_root_configs_compose_and_instantiate(tmp_path):
    fixture = tmp_path / "smoke.root"
    generate_test_root.generate_root(fixture, n_events=200)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        tt = compose(config_name="train.yaml", overrides=["experiment=orbit_root_tt_vq4096"])
        smoke = compose(
            config_name="train.yaml",
            overrides=["experiment=orbit_root_smoke", f"data.root_catalog.TT={fixture}"],
        )

    assert tt.data._target_ == "gabbro.data.orbit_canonical.CanonicalOrbitDataModule"
    assert tt.data.train_val_processes.TT.paths.endswith("TT_1760944672/data/merged.root")
    assert dict(tt.data.train_event_quotas) == {"TT": 188_000}
    assert dict(tt.data.val_event_quotas) == {"TT": 47_000}
    assert tt.data.test_suites.ttbar.event_budget == 59_000
    assert tt.data.sequence_type == "particle"
    assert tt.data.max_sequence_length == 128
    assert tt.data.root.sort_by_pt is True
    assert tt.data.root.split_block_size == 1000
    assert tt.data.pin_memory is False
    assert tt.model.model_kwargs.alpha == 10
    assert tt.model.optimizer.lr == 1e-4
    assert tt.model.scheduler is None
    assert tt.model.model_kwargs.vq_kwargs.num_codes == 4096

    dm = hydra.utils.instantiate(smoke.data)
    for loader in (dm.train_dataloader(), dm.val_dataloader(), *dm.test_dataloader()):
        batch = next(iter(loader))
        assert batch["part_features"].shape[1:] == (128, 4)
        assert batch["part_pid"].shape[1] == 128


def test_pin_memory_can_be_disabled(tmp_path):
    dm = _canonical(tmp_path, pin_memory=False)
    assert dm.train_dataloader().pin_memory is False
    assert all(loader.pin_memory is False for loader in dm.test_dataloader())
    default = _canonical(tmp_path)
    assert default.val_dataloader().pin_memory is torch.cuda.is_available()
