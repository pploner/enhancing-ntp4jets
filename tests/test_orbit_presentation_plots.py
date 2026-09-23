"""Tests for the cached canonical-tt presentation plot workflow."""

import csv

import numpy as np
import pytest

pytest.importorskip("mplhep")

from scripts.build_orbit_presentation_plots import (
    MAX_CODEBOOK_MARKER_AREA,
    MIN_CODEBOOK_MARKER_AREA,
    codebook_marker_areas,
    observable_values,
    select_plot_families,
    load_higgs_records,
    plot_mass_response,
    plot_mu_sigma,
)
from scripts.evaluate_orbit_higgs_mass import (
    mass_distribution_figure,
    plot_event_observables,
    plot_masses,
)


def _write_candidates(path, original, decoded):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("resolved_original", "resolved_decoded")
        )
        writer.writeheader()
        for original_value, decoded_value in zip(original, decoded):
            writer.writerow(
                {
                    "resolved_original": original_value,
                    "resolved_decoded": decoded_value,
                }
            )


def test_higgs_cache_uses_one_inclusive_reference(tmp_path):
    rng = np.random.default_rng(4)
    original = rng.normal(105, 12, 300)
    manifest = []
    for index, shift in enumerate((0.0, 2.0)):
        _write_candidates(
            tmp_path / f"{index:03d}" / "higgs_candidates.csv",
            original,
            original + shift,
        )
        manifest.append(
            {
                "label": f"run-{index}",
                "plot_family": "VQ STE",
                "run_dir": f"/run/{index}",
                "total_codebook_size": 128 * 2**index,
            }
        )

    records, reference_values, reference_fit = load_higgs_records(
        manifest, tmp_path, replicas=2
    )

    assert len(records) == 2
    assert np.array_equal(reference_values, original)
    assert reference_fit["success"]


def test_higgs_cache_rejects_inconsistent_original_candidates(tmp_path):
    original = np.linspace(80, 130, 100)
    for index, offset in enumerate((0, 1)):
        _write_candidates(
            tmp_path / f"{index:03d}" / "higgs_candidates.csv",
            original + offset,
            original,
        )
    manifest = [
        {
            "label": f"run-{index}",
            "run_dir": f"/run/{index}",
            "total_codebook_size": 128 * 2**index,
        }
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="shared reference"):
        load_higgs_records(manifest, tmp_path, replicas=1)


def test_mu_sigma_plot_is_title_free(tmp_path):
    records = [
        {
            "label": "vq-128",
            "plot_family": "VQ STE",
            "total_codebook_size": 128,
            "fit_success": True,
            "fit_mean": 105.0,
            "fit_sigma": 14.0,
            "fit_mean_uncertainty": 0.4,
            "fit_sigma_uncertainty": 0.6,
        }
    ]
    reference = {
        "success": True,
        "mean": 106.0,
        "sigma": 12.0,
        "mean_uncertainty": 0.3,
        "sigma_uncertainty": 0.5,
    }

    output = tmp_path / "mu_sigma.png"
    plot_mu_sigma(records, reference, output)

    assert output.is_file()


def test_higgs_mass_response_plot_uses_fitted_peak_ratio(tmp_path):
    records = [
        {
            "label": "vq-128",
            "plot_family": "VQ STE",
            "total_codebook_size": 128,
            "marginal_bits_per_input_particle": 8.0,
            "fit_success": True,
            "fit_mean": 100.0,
        },
        {
            "label": "vq-4096",
            "plot_family": "VQ STE",
            "total_codebook_size": 4096,
            "marginal_bits_per_input_particle": 12.0,
            "fit_success": True,
            "fit_mean": 110.0,
        },
    ]
    output = tmp_path / "mass_response.png"

    plot_mass_response(records, {"success": True, "mean": 125.0}, output)

    assert output.is_file()


def test_codebook_marker_area_increases_logarithmically():
    records = [
        {"total_codebook_size": 128},
        {"total_codebook_size": 1024},
        {"total_codebook_size": 16384},
    ]

    areas = codebook_marker_areas(records)

    assert areas[0] == pytest.approx(MIN_CODEBOOK_MARKER_AREA)
    assert areas[-1] == pytest.approx(MAX_CODEBOOK_MARKER_AREA)
    assert areas[0] < areas[1] < areas[2]


def test_faiss_vq_ste_family_filter_has_only_requested_families():
    records = [
        {"plot_family": "FAISS k-means"},
        {"plot_family": "VQ STE"},
        {"plot_family": "FSQ μ-only"},
    ]

    selected = select_plot_families(records, {"FAISS k-means", "VQ STE"})

    assert {record["plot_family"] for record in selected} == {
        "FAISS k-means",
        "VQ STE",
    }


def test_single_higgs_presentation_style_is_title_free_with_large_legend():
    original = np.linspace(75, 135, 200)
    decoded = original + 1

    figure = mass_distribution_figure(
        original,
        decoded,
        topology="resolved",
        candidate_mode="leading_pt",
        presentation=True,
    )
    axis = figure.axes[0]

    assert axis.get_title() == ""
    assert axis.get_legend().get_texts()[0].get_fontsize() == 12
    assert axis.patches[0].get_alpha() == pytest.approx(0.35)


def test_single_higgs_plot_accepts_extended_mass_bins():
    bins = np.arange(40, 505, 5)
    figure = mass_distribution_figure(
        np.linspace(50, 490, 200),
        np.linspace(55, 495, 200),
        presentation=True,
        bins=bins,
    )

    assert figure.axes[0].get_xlim()[1] >= 500


def test_single_higgs_plot_saves_histogram_cache(tmp_path):
    rows = [
        {"resolved_original": 100.0, "resolved_decoded": 101.0},
        {"resolved_original": 110.0, "resolved_decoded": 109.0},
    ]

    plot_masses(rows, "resolved", tmp_path, candidate_mode="leading_pt")

    with np.load(tmp_path / "higgs_mass_resolved_histograms.npz") as cache:
        assert np.array_equal(cache["bins"], np.arange(40, 205, 5))
        assert np.array_equal(cache["original"], [100.0, 110.0])
        assert np.array_equal(cache["decoded"], [101.0, 109.0])


def test_higgs_event_observable_plots_and_cache(tmp_path):
    rows = [
        {
            "leading_particle_pt_original": 80.0,
            "leading_particle_pt_decoded": 78.0,
            "n_jets_original": 3,
            "n_jets_decoded": 2,
        },
        {
            "leading_particle_pt_original": 120.0,
            "leading_particle_pt_decoded": 115.0,
            "n_jets_original": 4,
            "n_jets_decoded": 4,
        },
    ]

    plot_event_observables(rows, tmp_path)

    assert (tmp_path / "leading_particle_pt.png").is_file()
    assert (tmp_path / "n_jets.png").is_file()
    with np.load(tmp_path / "n_jets_histograms.npz") as cache:
        assert np.array_equal(cache["original"], [3, 4])
        assert np.array_equal(cache["decoded"], [2, 4])


def test_observable_values_reports_stale_higgs_cache(tmp_path):
    path = tmp_path / "higgs_candidates.csv"
    _write_candidates(path, [100.0], [101.0])

    with pytest.raises(ValueError, match="predates leading_particle_pt"):
        observable_values(path, "leading_particle_pt")
