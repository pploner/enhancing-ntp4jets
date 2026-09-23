"""Tests for additive, group-balanced canonical ORBIT experiments."""

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gabbro.data.orbit_parquet import (
    CanonicalOrbitParquetDataModule,
    OrbitParquetDataset,
    balanced_group_process_quotas,
    resolve_process_quotas,
)


def _write_particle_fixture(path, events=16):
    pq.write_table(
        pa.table(
            {
                "L1T_PUPPIPart_Eta": [[0.1, 0.2]] * events,
                "L1T_PUPPIPart_Phi": [[0.0, 1.0]] * events,
                "L1T_PUPPIPart_PT": [[10.0, 20.0]] * events,
                "L1T_PUPPIPart_PuppiW": [[1.0, 1.0]] * events,
            }
        ),
        path,
    )


def _count_labels(loader, key):
    counts = Counter()
    for batch in loader:
        counts.update(batch[key].tolist())
    return counts


def test_balanced_quotas_split_groups_before_processes():
    specs = {
        "qcd_a": {"group": "QCD"},
        "qcd_b": {"group": "QCD"},
        "tt_a": {"group": "tt"},
        "tt_b": {"group": "tt"},
        "tt_c": {"group": "tt"},
    }

    quotas = balanced_group_process_quotas(specs, total_events=11)

    assert quotas == {"qcd_a": 3, "qcd_b": 3, "tt_a": 2, "tt_b": 2, "tt_c": 1}
    assert sum(quotas.values()) == 11


def test_explicit_quotas_preserve_natural_process_counts():
    specs = {"hadronic": {}, "leptonic": {}, "semileptonic": {}}
    quotas = resolve_process_quotas(
        specs,
        event_budget=12,
        explicit_quotas={"hadronic": 3, "leptonic": 4, "semileptonic": 5},
        context="ttbar test",
    )

    assert quotas == {"hadronic": 3, "leptonic": 4, "semileptonic": 5}


@pytest.mark.parametrize(
    "explicit_quotas, message",
    [
        ({"hadronic": 5, "leptonic": 7}, "do not match processes"),
        (
            {"hadronic": 3, "leptonic": 4, "semileptonic": 4},
            "not event_budget=12",
        ),
    ],
)
def test_explicit_quotas_are_strictly_validated(explicit_quotas, message):
    specs = {"hadronic": {}, "leptonic": {}, "semileptonic": {}}
    with pytest.raises(ValueError, match=message):
        resolve_process_quotas(
            specs,
            event_budget=12,
            explicit_quotas=explicit_quotas,
            context="ttbar test",
        )


def test_parquet_metadata_is_scanned_lazily_until_event_budget(tmp_path, monkeypatch):
    paths = []
    for file_index in range(4):
        path = tmp_path / f"events_{file_index}.parquet"
        _write_particle_fixture(path, events=8)
        paths.append(str(path))

    opened = []
    parquet_file = pq.ParquetFile

    def tracking_parquet_file(path, *args, **kwargs):
        opened.append(str(path))
        return parquet_file(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", tracking_parquet_file)
    dataset = OrbitParquetDataset(
        paths,
        batch_size=4,
        max_events=4,
        max_sequence_length=4,
    )

    assert opened == []
    batches = list(dataset)
    assert sum(batch["part_features"].shape[0] for batch in batches) == 4
    assert opened == [paths[0]]


def test_canonical_module_emits_exact_quotas_and_named_test_suites(tmp_path):
    train_val_paths = {}
    test_paths = {}
    for process in ("qcd_a", "qcd_b", "tt_a"):
        process_train_val = []
        for file_index in range(2):
            path = tmp_path / f"{process}_train_val_{file_index}.parquet"
            _write_particle_fixture(path)
            process_train_val.append(str(path))
        train_val_paths[process] = process_train_val

        path = tmp_path / f"{process}_test.parquet"
        _write_particle_fixture(path)
        test_paths[process] = str(path)

    signal_path = tmp_path / "gghbb_test.parquet"
    _write_particle_fixture(signal_path)

    module = CanonicalOrbitParquetDataModule(
        train_val_processes={
            "qcd_a": {"paths": train_val_paths["qcd_a"], "group": "QCD"},
            "qcd_b": {"paths": train_val_paths["qcd_b"], "group": "QCD"},
            "tt_a": {"paths": train_val_paths["tt_a"], "group": "tt"},
        },
        test_suites={
            "training_like": {
                "event_budget": 8,
                "processes": {
                    "qcd_a": {"paths": test_paths["qcd_a"], "group": "QCD"},
                    "qcd_b": {"paths": test_paths["qcd_b"], "group": "QCD"},
                    "tt_a": {"paths": test_paths["tt_a"], "group": "tt"},
                },
            },
            "tt_vs_gghbb": {
                "event_budget": 6,
                "processes": {
                    "tt_a": {"paths": test_paths["tt_a"], "group": "tt"},
                    "ggHbb": {"paths": str(signal_path), "group": "ggHbb"},
                },
            },
        },
        train_event_budget=11,
        val_event_budget=7,
        batch_size=4,
        num_workers=0,
        max_sequence_length=4,
        pid_cfg={"enabled": False},
    )

    assert _count_labels(module.train_dataloader(), "process_labels") == Counter(
        {0: 3, 1: 3, 2: 5}
    )
    assert _count_labels(module.val_dataloader(), "process_labels") == Counter(
        {0: 2, 1: 2, 2: 3}
    )

    training_like, benchmark = module.test_dataloader()
    assert _count_labels(training_like, "process_labels") == Counter({0: 2, 1: 2, 2: 4})
    assert _count_labels(training_like, "test_suite_labels") == Counter({0: 8})
    assert _count_labels(benchmark, "process_labels") == Counter({2: 3, 3: 3})
    assert _count_labels(benchmark, "test_suite_labels") == Counter({1: 6})

    summary = module.data_split_summary()
    assert summary["test_suite_to_label"] == {"training_like": 0, "tt_vs_gghbb": 1}
    assert sum(row["event_count"] for row in summary["rows"] if row["split"] == "train") == 11
    assert sum(row["event_count"] for row in summary["rows"] if row["split"] == "val") == 7
