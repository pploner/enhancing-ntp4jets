"""Focused tests for the cached ORBIT writeup plot workflow."""

import json

import numpy as np
import pytest

pytest.importorskip("mplhep")

from gabbro.plotting.writeup.summary import (
    _safe_ratio,
    add_writeup_header,
    missing_golden_metric_records,
    plot_golden_summary,
    plot_jet_residuals,
    plot_met,
    plot_pid_confusion,
    plot_pid_residuals,
    plot_vq_size_feature_sweep,
    select_vq_ste_size_records,
)
from scripts.evaluate_orbit_jet_residuals import split_quota, wrapped_delta_phi
from scripts.render_orbit_writeup import config_to_argv, run_all


def test_safe_ratio_masks_empty_reference_bins():
    ratio = _safe_ratio(np.array([2, 3]), np.array([1, 0]))
    assert ratio[0] == 2
    assert np.isnan(ratio[1])


def test_wrapped_delta_phi_crosses_branch_cut():
    result = wrapped_delta_phi(np.array([-np.pi + 0.1]), np.array([np.pi - 0.1]))
    assert result[0] == pytest.approx(0.2)


def test_balanced_quota_preserves_total():
    assert split_quota(10_000, 3) == [3334, 3333, 3333]
    assert sum(split_quota(10_000, 21)) == 10_000


def test_unified_config_converts_repeated_and_boolean_options(monkeypatch):
    monkeypatch.setenv("ORBIT_TEST_ROOT", "/tmp/orbit")
    assert config_to_argv(
        {
            "campaign_summary_root": ["${ORBIT_TEST_ROOT}/one", "/tmp/two"],
            "allow_incomplete": True,
            "unused": None,
        }
    ) == [
        "--campaign-summary-root",
        "/tmp/orbit/one",
        "--campaign-summary-root",
        "/tmp/two",
        "--allow-incomplete",
    ]


def test_unified_all_runs_in_dependency_order(tmp_path, monkeypatch):
    config = tmp_path / "writeup.yaml"
    config.write_text(
        "training_scale:\n"
        "  - output_dir: /tmp/scale-one\n"
        "  - output_dir: /tmp/scale-two\n"
        "summary:\n"
        "  output_dir: /tmp/summary\n"
        "model_comparison:\n"
        "  output_dir: /tmp/comparison\n"
    )
    calls = []

    from scripts import render_orbit_writeup as renderer

    monkeypatch.setattr(renderer.training_scale, "parse_args", lambda argv: None)
    monkeypatch.setattr(renderer.summary, "parse_args", lambda argv: None)
    monkeypatch.setattr(renderer.model_comparison, "parse_args", lambda argv: None)
    monkeypatch.setattr(
        renderer.training_scale,
        "main",
        lambda argv: calls.append(("training-scale", argv)),
    )
    monkeypatch.setattr(
        renderer.summary, "main", lambda argv: calls.append(("summary", argv))
    )
    monkeypatch.setattr(
        renderer.model_comparison,
        "main",
        lambda argv: calls.append(("model-comparison", argv)),
    )

    run_all(["--config", str(config)])

    assert [name for name, _ in calls] == [
        "training-scale",
        "training-scale",
        "summary",
        "model-comparison",
    ]


def test_vq_sweep_requires_all_eight_histogram_caches():
    records = [
        {"plot_family": "VQ STE", "total_codebook_size": size, "histogram_path": "/cache"}
        for size in (128, 256, 512, 1024, 2048, 4096, 8192)
    ]
    with pytest.raises(ValueError, match="16384"):
        select_vq_ste_size_records(records)


def test_golden_summary_reports_unplottable_rows():
    complete = {
        "label": "complete",
        "total_codebook_size": 128,
        "plot_metrics/reco_mse_total": 0.1,
        "metrics/utilization_total": 1.0,
        "metrics/rate/marginal_bits_per_input_particle": 7.0,
    }
    assert missing_golden_metric_records([complete]) == []
    assert missing_golden_metric_records(
        [complete, {"plot_family": "VQ STE", "total_codebook_size": 256}]
    ) == ["VQ STE, 256 codes"]


def test_pid_confusion_recomputes_row_normalization(tmp_path):
    path = tmp_path / "pid.json"
    path.write_text(json.dumps({"truth_labels": ["a", "b"], "counts": [[3, 1], [0, 2]]}))
    figure, counts, normalized = plot_pid_confusion(path)
    assert np.array_equal(counts, [[3, 1], [0, 2]])
    assert np.allclose(normalized.sum(axis=1), 1)
    assert figure._suptitle is None


def test_pid_confusion_accepts_durable_count_cache(tmp_path):
    path = tmp_path / "pid.npz"
    counts = np.eye(8, dtype=np.int64)
    counts[6, 6] = 3
    counts[6, 7] = 1
    np.savez_compressed(path, counts=counts)
    _, loaded, normalized = plot_pid_confusion(path)
    assert np.array_equal(loaded, counts)
    assert normalized[6, 6] == pytest.approx(0.75)
    assert normalized[6, 7] == pytest.approx(0.25)


def test_jet_residual_plot_is_title_free(tmp_path):
    path = tmp_path / "jets.npz"
    np.savez_compressed(
        path,
        all_delta_pt=np.linspace(-0.2, 0.2, 20),
        all_delta_eta=np.linspace(-0.1, 0.1, 20),
        all_delta_phi=np.linspace(-0.1, 0.1, 20),
    )
    figure, _ = plot_jet_residuals(path, "all")
    assert figure._suptitle is None
    assert len(figure.axes) == 3
    assert [axis.get_ylabel() for axis in figure.axes] == ["Normalized density", "", ""]
    assert all(
        not patch.get_fill() for axis in figure.axes for patch in axis.patches
    )


def test_met_plot_uses_cached_histograms_and_ratio(tmp_path):
    path = tmp_path / "met.npz"
    np.savez_compressed(
        path,
        physical_missing_et_bins=np.array([0.0, 20.0, 40.0]),
        physical_missing_et_orig_counts=np.array([0.02, 0.03]),
        physical_missing_et_reco_counts=np.array([0.018, 0.033]),
        physical_missing_et_diff_bins=np.array([-20.0, 0.0, 20.0]),
        physical_missing_et_diff_counts=np.array([0.01, 0.04]),
    )
    figure, values = plot_met(path)
    assert figure._suptitle is None
    assert len(figure.axes) == 3
    assert np.allclose(values["decoded_over_original"], [0.9, 1.1])
    assert figure.axes[1].get_xlim() == pytest.approx((0.0, 600.0))
    assert figure.axes[2].get_xlim() == pytest.approx((-200.0, 200.0))
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
        "Original",
        "Decoded",
    ]


def test_pid_residual_plot_uses_symbols_and_one_y_label(tmp_path):
    path = tmp_path / "pid.npz"
    values = np.linspace(-0.1, 0.1, 16)
    np.savez_compressed(
        path,
        original_pid=np.arange(16) % 8,
        delta_eta=values,
        delta_phi=values,
        delta_log_pt=values,
    )
    figure = plot_pid_residuals(path)
    assert [axis.get_ylabel() for axis in figure.axes] == ["Normalized density", "", ""]
    assert [text.get_text() for text in figure.axes[-1].get_legend().get_texts()] == [
        r"$h^0$",
        r"$\gamma$",
        r"$h^-$",
        r"$h^+$",
        r"$e^-$",
        r"$e^+$",
        r"$\mu^-$",
        r"$\mu^+$",
    ]


def test_golden_summary_is_one_row():
    records = [
        {
            "label": "run",
            "plot_family": "VQ STE",
            "total_codebook_size": 4096,
            "plot_metrics/reco_mse_total": 0.1,
            "metrics/utilization_total": 0.8,
            "metrics/rate/marginal_bits_per_input_particle": 7.0,
        }
    ]
    figure = plot_golden_summary(records)
    assert len(figure.axes) == 3
    assert len({axis.get_position().y0 for axis in figure.axes}) == 1
    assert max(text.get_fontsize() for text in figure.axes[-1].get_legend().get_texts()) <= 12


def test_vq_feature_sweep_uses_compact_legend_and_residual_row(tmp_path):
    bins = np.linspace(-1, 1, 5)
    counts = np.ones(4)
    records = []
    for size in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        path = tmp_path / f"{size}.npz"
        payload = {}
        for feature in ("physical_pT", "physical_Eta", "physical_Phi"):
            payload[f"{feature}_bins"] = bins
            payload[f"{feature}_orig_counts"] = counts
            payload[f"{feature}_reco_counts"] = counts
            payload[f"{feature}_diff_bins"] = bins
            payload[f"{feature}_diff_counts"] = counts
        np.savez_compressed(path, **payload)
        records.append(
            {
                "plot_family": "VQ STE",
                "total_codebook_size": size,
                "histogram_path": str(path),
            }
        )

    figure = plot_vq_size_feature_sweep(records)
    legend = figure.axes[0].get_legend()
    assert max(text.get_fontsize() for text in legend.get_texts()) <= 14
    assert figure.axes[2].get_subplotspec().get_gridspec().get_height_ratios() == (
        1.45,
        0.9,
    )

    top_axes = (figure.axes[0], figure.axes[3], figure.axes[6])
    add_writeup_header(
        figure,
        top_axes,
        training_text=r"$t\bar{t}$ (200k)",
        test_text=r"$t\bar{t}$ (10k)",
        selection_text=r"Pileup-subtracted  |  $w_{\mathrm{PUPPI}}>0.05$  |  $N_{\max}=128$",
    )
    figure.canvas.draw()
    panel_tops = [axis.get_position().y1 for axis in top_axes]
    assert max(panel_tops) - min(panel_tops) < 1e-3
    assert all("CMS" not in text.get_text() for text in figure.texts)
    assert all("Simulation" not in text.get_text() for text in figure.texts)
    assert all("TeV" not in text.get_text() for text in figure.texts)
