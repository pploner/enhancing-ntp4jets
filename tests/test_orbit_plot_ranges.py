"""Regression tests for shared ORBIT validation/test plot conventions."""

import matplotlib.pyplot as plt
import numpy as np

from gabbro.plotting.orbit import (
    collect_physical_reconstruction_histograms,
    collect_reconstruction_histograms,
    plot_feature_histograms,
    physical_reconstruction_plots,
)


def test_class_specific_physics_ranges_and_overflow_bins():
    features = ["Eta", "Phi", "pT"]
    original = np.array([[0.2, 0.1, 20.0], [0.3, -0.2, 30.0]])
    histograms = collect_physical_reconstruction_histograms(
        features,
        original,
        original,
        true_jet_masses=np.array([50.0, 4_000.0]),
        reco_jet_masses=np.array([60.0, 4_100.0]),
        true_tau32s=np.array([0.3, 0.4]),
        reco_tau32s=np.array([0.2, 0.5]),
        true_missing_ets=np.array([10.0, 300.0]),
        reco_missing_ets=np.array([20.0, 400.0]),
        missing_et_range=(0.0, 200.0),
        jet_mass_range=(0.0, 3_000.0),
    )

    assert (histograms["missing_et_bins"][0], histograms["missing_et_bins"][-1]) == (
        0.0,
        200.0,
    )
    assert (
        histograms["missing_et_diff_bins"][0],
        histograms["missing_et_diff_bins"][-1],
    ) == (-200.0, 200.0)
    assert (histograms["jet_mass_bins"][0], histograms["jet_mass_bins"][-1]) == (
        0.0,
        3_000.0,
    )
    assert (
        histograms["energy_residuals_bins"][0],
        histograms["energy_residuals_bins"][-1],
    ) == (-50.0, 50.0)

    figures = physical_reconstruction_plots(
        features,
        np.zeros(3),
        histograms,
        "particle",
    )
    assert figures["jet_substructure"].axes[0].get_yscale() == "log"
    plt.close("all")


def test_transformed_pt_histogram_uses_log_density_axis():
    features = ["L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi_cos", "L1T_PUPPIPart_PT"]
    values = np.array([[[0.1, 0.2, 1.0], [0.2, -0.1, 2.0]]])
    histograms = collect_reconstruction_histograms(
        features,
        values,
        values,
        np.ones((1, 2), dtype=bool),
    )

    figure = plot_feature_histograms(histograms, features)
    assert figure.axes[2].get_yscale() == "log"
    plt.close(figure)
