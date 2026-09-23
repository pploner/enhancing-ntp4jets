"""Focused tests for paired downstream physics-fidelity helpers."""

import json
import runpy

import awkward as ak
import numpy as np
import pytest
from omegaconf import OmegaConf

from gabbro.data.orbit_downstream import (
    FIVE_CLASS_GROUPS,
    FIVE_CLASS_TO_LABEL,
    PairedOrbitDataModule,
    physical_particles_to_classifier_features,
)
from gabbro.models.orbit_event_classifier import classification_metrics


@pytest.mark.parametrize(
    "script",
    [
        "scripts/evaluate_orbit_higgs_mass.py",
        "scripts/evaluate_orbit_z_mumu_mass.py",
    ],
)
def test_mass_benchmark_inherits_checkpoint_energy_settings(script):
    namespace = runpy.run_path(script)
    settings = namespace["checkpoint_energy_settings"]

    assert settings(OmegaConf.create({"data": {}})) == (False, 2.5)
    assert settings(
        OmegaConf.create({"data": {"include_energy": True, "energy_shift": 2.7}})
    ) == (True, 2.7)


def test_five_class_taxonomy_and_feature_contract():
    assert list(FIVE_CLASS_GROUPS) == ["QCD", "tt", "VJets", "VV", "ggHbb"]
    assert [len(processes) for processes in FIVE_CLASS_GROUPS.values()] == [2, 3, 7, 9, 1]
    assert FIVE_CLASS_TO_LABEL["ggHbb"] == 4

    particles = ak.Array([[[0.3, np.pi / 2, 10.0], [-0.6, 0.0, 20.0]]])
    pid = ak.Array([[1, 7]])
    features, mask = physical_particles_to_classifier_features(particles, pid, 4)
    assert features.shape == (1, 4, 12)
    assert mask.tolist() == [[True, True, False, False]]
    assert np.allclose(
        features[0, 0, :4], [0.1, 0.0, 1.0, np.log(10.0) - 1.8], atol=1e-6
    )
    assert features[0, 0, 5] == 1
    assert features[0, 1, 11] == 1
    assert np.count_nonzero(features[0, 2:].numpy()) == 0


def test_classification_metrics_are_perfect_for_perfect_logits():
    targets = np.repeat(np.arange(5), 3)
    logits = np.full((len(targets), 5), -10.0)
    logits[np.arange(len(targets)), targets] = 10.0
    metrics = classification_metrics(targets, logits)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_auroc"] == 1.0
    assert np.array_equal(metrics["confusion_matrix"], np.eye(5, dtype=int) * 3)


def test_paired_data_module_derives_full_event_length_and_batch_size(tmp_path):
    metadata = {
        "class_to_label": FIVE_CLASS_TO_LABEL,
        "sequence_type": "particle_full",
        "max_sequence_length": 500,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    module = PairedOrbitDataModule(str(tmp_path), num_workers=0)
    assert module.hparams.max_sequence_length == 500
    assert module.hparams.batch_size == 16

    with pytest.raises(ValueError, match="does not match paired data"):
        PairedOrbitDataModule(str(tmp_path), max_sequence_length=128, num_workers=0)


def test_higgs_truth_selector_uses_direct_bb_daughters():
    namespace = runpy.run_path("scripts/evaluate_orbit_higgs_mass.py")
    truth = ak.Record(
        {
            "Gen_Part_PT": [100.0, 110.0, 50.0, 45.0],
            "Gen_Part_Eta": [0.0, 0.1, 0.2, -0.2],
            "Gen_Part_Phi": [0.0, 0.1, 0.2, -0.2],
            "Gen_Part_Mass": [125.0, 125.0, 4.8, 4.8],
            "Gen_Part_PID": [25, 25, 5, -5],
            "Gen_Part_D1": [-1, 2, -1, -1],
            "Gen_Part_D2": [-1, 3, -1, -1],
        }
    )
    higgs, b, bbar = namespace["decaying_higgs"](truth)
    assert higgs[2] == 110.0
    assert {b[2], bbar[2]} == {45.0, 50.0}


def test_higgs_candidates_are_stable_for_perfect_reconstruction():
    namespace = runpy.run_path("scripts/evaluate_orbit_higgs_mass.py")
    particles = np.array(
        [
            [0.10, 0.10, 180.0],
            [0.12, 0.12, 80.0],
            [-0.35, -0.30, 170.0],
            [-0.38, -0.28, 70.0],
        ]
    )
    b = np.array([0.1, 0.1, 200.0, 4.8])
    bbar = np.array([-0.35, -0.3, 190.0, 4.8])
    higgs = np.array([-0.1, -0.1, 390.0, 125.0])
    resolved = namespace["resolved_candidate"](particles, b, bbar)
    boosted = namespace["boosted_candidate"](particles, higgs, b, bbar)
    assert resolved is not None
    assert boosted is not None
    assert resolved == namespace["resolved_candidate"](particles.copy(), b, bbar)
    assert boosted == namespace["boosted_candidate"](particles.copy(), higgs, b, bbar)


def test_higgs_leading_pt_and_cross_hungarian_candidates_are_truth_free():
    namespace = runpy.run_path("scripts/evaluate_orbit_higgs_mass.py")
    original = np.array(
        [[0.0, 0.0, 100.0], [1.0, 1.0, 90.0], [-2.0, -2.0, 40.0]]
    )
    decoded = np.array(
        [[0.02, 0.01, 50.0], [1.02, 0.99, 45.0], [-2.5, -2.5, 200.0]]
    )
    leading = namespace["leading_higgs_candidate"](original)
    assert leading is not None
    assert np.allclose(leading["jets"][:, 2], [100.0, 90.0])

    original_candidate, decoded_candidate = namespace["cross_matched_higgs_candidates"](
        original, decoded
    )
    assert original_candidate is not None
    assert decoded_candidate is not None
    assert np.all(decoded_candidate["match_dr"] < 0.05)
    assert np.allclose(decoded_candidate["jets"][:, 2], [50.0, 45.0])
    _, rejected = namespace["cross_matched_higgs_candidates"](
        original, decoded, max_match_dr=0.01
    )
    assert rejected is None


def test_higgs_multirun_plot_overlays_decoded_outlines(tmp_path):
    namespace = runpy.run_path("scripts/evaluate_orbit_higgs_mass.py")
    rows = [
        {"resolved_original": 120.0, "resolved_decoded": 118.0, "boosted_original": 125.0, "boosted_decoded": 123.0},
        {"resolved_original": 130.0, "resolved_decoded": 128.0, "boosted_original": 135.0, "boosted_decoded": 133.0},
    ]
    namespace["plot_multirun_masses"](
        [("reference", rows, {}), ("comparison", rows, {})], "resolved", tmp_path
    )
    assert (tmp_path / "higgs_mass_resolved_multirun.png").is_file()


def test_z_truth_selector_requires_direct_opposite_sign_muon_daughters():
    namespace = runpy.run_path("scripts/evaluate_orbit_z_mumu_mass.py")
    truth = ak.Record(
        {
            "Gen_Part_PT": [100.0, 125.0, 45.0, 42.0],
            "Gen_Part_Eta": [0.0, 0.2, 0.3, -0.3],
            "Gen_Part_Phi": [0.0, 0.4, 0.5, -0.5],
            "Gen_Part_Mass": [91.0, 91.2, 0.105, 0.105],
            "Gen_Part_PID": [23, 23, 13, -13],
            "Gen_Part_D1": [-1, 2, -1, -1],
            "Gen_Part_D2": [-1, 3, -1, -1],
        }
    )
    z = namespace["decaying_z_to_mumu"](truth)
    assert z["z"][2] == 125.0
    assert z["muon"][2] == 45.0
    assert z["antimuon"][2] == 42.0
    non_dimuon_truth = ak.Record(
        {
            "Gen_Part_PT": [100.0, 45.0, 42.0],
            "Gen_Part_Eta": [0.0, 0.3, -0.3],
            "Gen_Part_Phi": [0.0, 0.5, -0.5],
            "Gen_Part_Mass": [91.0, 0.105, 0.105],
            "Gen_Part_PID": [23, 11, -11],
            "Gen_Part_D1": [1, -1, -1],
            "Gen_Part_D2": [2, -1, -1],
        }
    )
    assert namespace["decaying_z_to_mumu"](non_dimuon_truth) is None


def test_z_dimuon_candidate_matches_direct_truth_daughters_with_hungarian_assignment():
    namespace = runpy.run_path("scripts/evaluate_orbit_z_mumu_mass.py")
    particles = np.array(
        [
            [0.10, 0.10, 18.0],
            [0.20, 0.20, 52.0],
            [-0.30, -2.90, 47.0],
            [-0.20, -2.80, 20.0],
        ]
    )
    pid = np.array([6, 6, 7, 7])
    candidate = namespace["truth_matched_dimuon_candidate"](
        particles,
        pid,
        truth_muon=np.array([0.20, 0.20, 50.0, 0.105]),
        truth_antimuon=np.array([-0.30, -2.90, 45.0, 0.105]),
    )
    assert candidate["muon_index"] == 1
    assert candidate["antimuon_index"] == 2
    assert candidate["muon_dr"] == 0.0
    assert candidate["antimuon_dr"] == 0.0
    assert candidate["mass"] > 0


def test_z_dimuon_candidate_requires_pid_and_delta_r_match():
    namespace = runpy.run_path("scripts/evaluate_orbit_z_mumu_mass.py")
    particles = np.array([[0.01, 0.01, 45.0], [-0.01, -0.01, 44.0]])
    truth_muon = np.array([0.0, 0.0, 45.0, 0.105])
    truth_antimuon = np.array([0.0, 0.0, 44.0, 0.105])
    matcher = namespace["truth_matched_dimuon_candidate"]
    assert matcher(particles, np.array([3, 7]), truth_muon, truth_antimuon) is None
    assert matcher(
        np.array([[0.4, 0.0, 45.0], [0.0, 0.4, 44.0]]),
        np.array([6, 7]),
        truth_muon,
        truth_antimuon,
    ) is None


def test_z_leading_pt_and_cross_hungarian_candidates_are_pid_constrained():
    namespace = runpy.run_path("scripts/evaluate_orbit_z_mumu_mass.py")
    original = np.array(
        [[0.0, 0.0, 60.0], [0.8, 2.8, 55.0], [0.1, 0.1, 20.0]]
    )
    original_pid = np.array([6, 7, 6])
    decoded = np.array(
        [[0.02, 0.01, 30.0], [0.82, 2.79, 28.0], [-2.5, -2.5, 100.0]]
    )
    decoded_pid = np.array([6, 7, 6])
    leading = namespace["leading_dimuon_candidate"](original, original_pid)
    assert leading["muon_index"] == 0
    assert leading["antimuon_index"] == 1

    original_candidate, decoded_candidate = namespace["cross_matched_dimuon_candidates"](
        original, original_pid, decoded, decoded_pid
    )
    assert original_candidate is not None
    assert decoded_candidate is not None
    assert decoded_candidate["muon_index"] == 0
    assert decoded_candidate["antimuon_index"] == 1
    assert decoded_candidate["muon_pid"] == 6
    assert decoded_candidate["antimuon_pid"] == 7
    _, rejected = namespace["cross_matched_dimuon_candidates"](
        original, original_pid, decoded, decoded_pid, max_match_dr=0.01
    )
    assert rejected is None


def test_optional_eta_acceptance_is_separate_from_object_existence():
    higgs = runpy.run_path("scripts/evaluate_orbit_higgs_mass.py")
    particles = np.array([[3.0, 0.0, 100.0], [0.0, 2.0, 90.0]])
    assert higgs["leading_higgs_candidate"](particles) is not None
    assert higgs["leading_higgs_candidate"](particles, max_abs_eta=2.5) is None

    z = runpy.run_path("scripts/evaluate_orbit_z_mumu_mass.py")
    pid = np.array([6, 7])
    assert z["leading_dimuon_candidate"](particles, pid) is not None
    assert z["leading_dimuon_candidate"](particles, pid, max_abs_eta=2.5) is None


def test_z_multirun_plot_overlays_decoded_outlines(tmp_path):
    namespace = runpy.run_path("scripts/evaluate_orbit_z_mumu_mass.py")
    rows = [
        {"original_mass": 90.0, "decoded_mass": 89.0},
        {"original_mass": 92.0, "decoded_mass": 93.0},
    ]
    namespace["plot_multirun_masses"](
        [("reference", rows, {}), ("comparison", rows, {})], tmp_path
    )
    assert (tmp_path / "z_mumu_mass_multirun.png").is_file()


def test_mass_benchmarks_report_empirical_distribution_moments():
    for script in (
        "scripts/evaluate_orbit_higgs_mass.py",
        "scripts/evaluate_orbit_z_mumu_mass.py",
    ):
        namespace = runpy.run_path(script)
        moments = namespace["distribution_moments"]([80.0, 100.0, np.nan])
        assert moments == {"events": 2, "mean": 90.0, "std": 10.0}
        label = namespace["distribution_label"]("Decoded", [80.0, 100.0])
        assert "Decoded" in label
        assert "90.0" in label
        assert "10.0" in label
