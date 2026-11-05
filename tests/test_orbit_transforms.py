"""Configurable ORBIT kinematic preprocessing (data.eta_scale / data.pt_shift)."""

from types import SimpleNamespace

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from gabbro.callbacks.orbit_plotting_callback import OrbitPlottingCallback
from gabbro.data.orbit_common import OrbitPreprocessor
from gabbro.data.orbit_parquet import CanonicalOrbitParquetDataModule, OrbitParquetDataset
from gabbro.data.orbit_transforms import (
    DEFAULT_ETA_SCALE,
    DEFAULT_PT_SHIFT,
    KinematicTransform,
)
from gabbro.plotting.orbit import pid_residual_arrays

ETA = [[0.1, -2.4, 1.3], [3.9, -0.7]]
PHI = [[0.0, 2.0, -1.5], [3.0, -3.0]]
PT = [[1.0, 7.25, 120.0], [2.5, 35.0]]


def _write_fixture(path, events=4):
    pq.write_table(
        pa.table(
            {
                "L1T_PUPPIPart_Eta": ETA * (events // 2),
                "L1T_PUPPIPart_Phi": PHI * (events // 2),
                "L1T_PUPPIPart_PT": PT * (events // 2),
                "L1T_PUPPIPart_PuppiW": [[1.0] * len(row) for row in ETA] * (events // 2),
            }
        ),
        path,
    )


@pytest.mark.parametrize(
    "config",
    [None, {}, OmegaConf.create({"eta_scale": None}), SimpleNamespace(include_energy=False)],
)
def test_missing_config_falls_back_to_historical_constants(config):
    transform = KinematicTransform.from_data_config(config)
    assert (transform.eta_scale, transform.pt_shift) == (3.0, 1.8)
    assert (DEFAULT_ETA_SCALE, DEFAULT_PT_SHIFT) == (3.0, 1.8)


def test_config_values_are_read_from_mappings_and_hparams():
    for config in (
        {"eta_scale": 1.7, "pt_shift": 0.6},
        OmegaConf.create({"eta_scale": 1.7, "pt_shift": 0.6}),
    ):
        assert KinematicTransform.from_data_config(config) == KinematicTransform(1.7, 0.6)


@pytest.mark.parametrize("eta_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_eta_scale_is_rejected(eta_scale):
    with pytest.raises(ValueError, match="eta_scale"):
        KinematicTransform(eta_scale=eta_scale)


def test_default_preprocessor_matches_historical_formula_bitwise():
    array = ak.zip(
        {
            "L1T_PUPPIPart_Eta": ak.values_astype(ak.Array(ETA), np.float32),
            "L1T_PUPPIPart_Phi": ak.values_astype(ak.Array(PHI), np.float32),
            "L1T_PUPPIPart_PT": ak.values_astype(ak.Array(PT), np.float16),
        }
    )
    out = OrbitPreprocessor("L1T_PUPPIPart").forward(array)
    expected_eta = array["L1T_PUPPIPart_Eta"] / 3
    expected_pt = np.log(array["L1T_PUPPIPart_PT"] + 1e-8) - 1.8
    assert ak.to_list(out["L1T_PUPPIPart_Eta"]) == ak.to_list(expected_eta)
    assert ak.to_list(out["L1T_PUPPIPart_PT"]) == ak.to_list(expected_pt)


@pytest.mark.parametrize(
    "eta_scale, pt_shift",
    [(DEFAULT_ETA_SCALE, DEFAULT_PT_SHIFT), (1.0, 0.0), (1.7, 0.6), (3.0, 0.0), (1.0, 1.8)],
)
def test_dataset_features_round_trip_to_physical_units(tmp_path, eta_scale, pt_shift):
    path = tmp_path / "particles.parquet"
    _write_fixture(path)
    batch = next(
        iter(
            OrbitParquetDataset(
                [path],
                sequence_type="particle_full",
                max_sequence_length=3,
                batch_size=2,
                eta_scale=eta_scale,
                pt_shift=pt_shift,
            )
        )
    )
    features = batch["part_features"].numpy()
    mask = batch["part_mask"].numpy().astype(bool)
    assert np.allclose(features[0, :, 0], np.asarray(ETA[0]) / eta_scale, atol=1e-6)
    assert np.allclose(features[0, :, 3], np.log(PT[0]) - pt_shift, atol=1e-5)

    physical = KinematicTransform(eta_scale, pt_shift).to_physical(features)[mask]
    expected_phi = np.arctan2(np.sin(np.concatenate(PHI)), np.cos(np.concatenate(PHI)))
    assert np.allclose(physical[:, 0], np.concatenate(ETA), atol=1e-5)
    assert np.allclose(physical[:, 1], expected_phi, atol=1e-5)
    assert np.allclose(physical[:, 2], np.concatenate(PT), rtol=1e-5)


def test_datamodule_passes_transform_to_datasets_and_callback(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"tt_{index}.parquet"
        _write_fixture(path, events=8)
        paths.append(str(path))
    test_path = str(tmp_path / "tt_test.parquet")
    _write_fixture(test_path)
    module = CanonicalOrbitParquetDataModule(
        train_val_processes={"tt_a": {"paths": paths, "group": "tt"}},
        test_suites={
            "tt": {"event_budget": 2, "processes": {"tt_a": {"paths": test_path, "group": "tt"}}}
        },
        train_event_budget=4,
        val_event_budget=2,
        sequence_type="particle_full",
        max_sequence_length=3,
        batch_size=4,
        num_workers=0,
        pid_cfg={"enabled": False},
        eta_scale=1.5,
        pt_shift=0.25,
    )
    transform = KinematicTransform.from_data_config(module.hparams)
    assert transform == KinematicTransform(1.5, 0.25)

    batch = next(iter(module.train_dataloader()))
    features = batch["part_features"].numpy()
    mask = batch["part_mask"].numpy().astype(bool)
    assert np.allclose(np.abs(features[..., 0][mask]).max(), 3.9 / 1.5, atol=1e-6)

    trainer = SimpleNamespace(datamodule=module)
    callback = OrbitPlottingCallback.__new__(OrbitPlottingCallback)
    physical = callback._to_physical_features(trainer, features)
    assert np.allclose(physical[..., 0][mask], transform.eta_inverse(features[..., 0][mask]))
    assert np.allclose(np.sort(physical[..., 2][mask])[-1], 120.0, rtol=1e-4)


def test_pid_residual_delta_eta_uses_eta_scale():
    names = (
        "L1T_PUPPIPart_Eta",
        "L1T_PUPPIPart_Phi_cos",
        "L1T_PUPPIPart_Phi_sin",
        "L1T_PUPPIPart_PT",
    )
    original = np.array([[0.1, 1.0, 0.0, 1.0]])
    reconstructed = np.array([[0.3, 1.0, 0.0, 1.5]])
    default = pid_residual_arrays(original, reconstructed, names)
    unscaled = pid_residual_arrays(original, reconstructed, names, eta_scale=1.0)
    assert np.allclose(default["delta_eta"], 0.6)
    assert np.allclose(unscaled["delta_eta"], 0.2)
    assert np.allclose(default["delta_log_pt"], unscaled["delta_log_pt"])


def test_default_labels_are_unchanged_and_custom_labels_follow_config():
    from gabbro.plotting.utils import DEFAULT_LABELS, orbit_feature_label

    for name, label in DEFAULT_LABELS.items():
        if name.startswith("L1T_"):
            assert orbit_feature_label(name, KinematicTransform()) == label
            assert orbit_feature_label(name) == label
    assert orbit_feature_label("L1T_PUPPIPart_Eta", KinematicTransform(1.0, 0.0)) == (
        r"PUPPI particle $\eta$"
    )
    assert orbit_feature_label("L1T_PUPPIPart_PT", KinematicTransform(1.0, 0.0)) == (
        r"PUPPI particle $\log(p_T)$"
    )
    fitted = KinematicTransform(2.305123, -0.5)
    assert orbit_feature_label("L1T_JetAK4_Eta", fitted) == r"AK4 jet $\eta / 2.305$"
    assert orbit_feature_label("L1T_JetAK4_PT", fitted) == r"AK4 jet $\log(p_T) + 0.5$"
    assert (
        orbit_feature_label("L1T_PUPPIPart_Phi_cos", fitted)
        == (DEFAULT_LABELS["L1T_PUPPIPart_Phi_cos"])
    )


def test_multirun_comparison_refuses_to_overlay_mixed_preprocessing():
    pytest.importorskip("mplhep")
    from scripts.collect_orbit_multirun import _common_kinematics, _is_transformed_eta_or_pt

    assert _common_kinematics([{}, {"eta_scale": 3.0, "pt_shift": 1.8}]) == KinematicTransform()
    assert _common_kinematics([{"eta_scale": 1.0, "pt_shift": 0.0}] * 2) == KinematicTransform(
        1.0, 0.0
    )
    assert _common_kinematics([{}, {"eta_scale": 1.0}]) is None
    assert _is_transformed_eta_or_pt("L1T_PUPPIPart_Eta")
    assert _is_transformed_eta_or_pt("L1T_PUPPIPart_PT")
    assert not _is_transformed_eta_or_pt("physical_Eta")
    assert not _is_transformed_eta_or_pt("L1T_PUPPIPart_Phi_cos")
