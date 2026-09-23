"""Tests for entropy and rate-distortion multirun outputs."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mplhep")

from scripts.collect_orbit_multirun import (
    _artifact_dirs,
    _compression_ratio_figure,
    _parse_artifact_family_runs,
    _records_with_compression_ratio,
    _save_figures,
)
from scripts.collect_orbit_multirun import _quantizer_metadata
from scripts.collect_orbit_multirun import _wandb_upload_settings
from scripts.collect_orbit_multirun import _artifact_prefix
from gabbro.plotting.orbit import (
    multirun_color,
    multirun_display_label,
    multirun_legend_sort_key,
    multirun_marker,
    plot_multirun_metric,
)


def test_entropy_multirun_figures_are_written(tmp_path):
    records = []
    for label, codebook_size, entropy, rate, error in (
        ("small", 16, 3.0, 1.5, 0.10),
        ("large", 256, 6.0, 3.0, 0.05),
    ):
        records.append(
            {
                "label": label,
                "plot_family": "FSQ",
                "total_codebook_size": codebook_size,
                "metrics/entropy/marginal_bits_per_token": entropy,
                "metrics/entropy/normalized_to_log2_codebook": 0.75,
                "metrics/rate/marginal_bits_per_event": rate * 20,
                "metrics/rate/marginal_bits_per_input_particle": rate,
                "plot_metrics/reco_mse_total": error,
            }
        )

    _save_figures(records, histogram_runs=[], output_dir=tmp_path)

    expected = {
        "codebook_size_vs_marginal_entropy_bits_per_token.png",
        "codebook_size_vs_normalized_entropy.png",
        "codebook_size_vs_marginal_bits_per_event.png",
        "codebook_size_vs_marginal_bits_per_input_particle.png",
        "marginal_bits_per_input_particle_vs_reco_mse.png",
        "compression_ratio_40bit_vs_reco_mse.png",
        "marginal_bits_per_event_vs_reco_mse.png",
    }
    assert expected.issubset({path.name for path in tmp_path.glob("*.png")})


def test_compression_ratio_uses_40_bit_payload_denominator():
    records = [{"metrics/rate/marginal_bits_per_input_particle": 10.0}]

    normalized = _records_with_compression_ratio(records, reference_bits=40.0)

    assert normalized[0]["plot_metrics/compression_ratio"] == pytest.approx(0.25)
    assert "plot_metrics/compression_ratio" not in records[0]


def test_compression_ratio_plot_has_room_for_full_axis_label():
    figure = _compression_ratio_figure(
        [
            {
                "label": "VQ STE",
                "metrics/rate/marginal_bits_per_input_particle": 10.0,
                "plot_metrics/reco_mse_total": 0.1,
            }
        ]
    )

    assert tuple(figure.get_size_inches()) == (9.5, 6.0)


def test_compression_ratio_plot_is_linear_without_payload_reference_line():
    figure = _compression_ratio_figure(
        [
            {
                "label": "VQ STE",
                "metrics/rate/marginal_bits_per_input_particle": 10.0,
                "plot_metrics/reco_mse_total": 0.1,
            }
        ]
    )

    assert figure.axes[0].get_xscale() == "linear"
    assert "40-bit input payload" not in {
        line.get_label() for line in figure.axes[0].lines
    }


def test_compression_ratio_marker_area_increases_with_codebook_size():
    figure = _compression_ratio_figure(
        [
            {
                "label": "small",
                "plot_family": "VQ STE",
                "total_codebook_size": 128,
                "metrics/rate/marginal_bits_per_input_particle": 7.0,
                "plot_metrics/reco_mse_total": 0.2,
            },
            {
                "label": "large",
                "plot_family": "VQ STE",
                "total_codebook_size": 16384,
                "metrics/rate/marginal_bits_per_input_particle": 14.0,
                "plot_metrics/reco_mse_total": 0.1,
            },
        ]
    )

    marker_areas = figure.axes[0].collections[0].get_sizes()
    assert marker_areas[0] < marker_areas[1]
    assert figure.axes[0].get_legend().get_title().get_text() == ""


def test_faiss_kmeans_multirun_metadata():
    metadata = _quantizer_metadata(
        {
            "model": {
                "_target_": "gabbro.models.vqvae.FaissKMeansBaselineLightning",
                "num_codes": 8192,
            }
        }
    )

    assert metadata["quantizer_mode"] == "kmeans"
    assert metadata["quantizer_family"] == "kmeans"
    assert metadata["total_codebook_size"] == 8192


def test_named_test_suite_artifact_prefix():
    assert _artifact_prefix("test", "all", "training_like") == "test_training_like_all_orbit"
    assert _artifact_prefix("test", "tt", "tt_vs_gghbb") == "test_tt_vs_gghbb_tt_orbit"


def test_explicit_artifact_directory_takes_precedence(tmp_path):
    artifact_dir = tmp_path / "cross_test"
    artifact_dir.mkdir()

    assert _artifact_dirs(tmp_path / "run", artifact_dir) == [artifact_dir]


def test_artifact_family_parser_pairs_runs_with_artifacts(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    artifact_a = tmp_path / "artifact_a"
    artifact_b = tmp_path / "artifact_b"
    for path in (run_a, run_b):
        (path / ".hydra").mkdir(parents=True)
        (path / ".hydra" / "config.yaml").touch()
    artifact_a.mkdir()
    artifact_b.mkdir()

    assert _parse_artifact_family_runs(
        [["trained on tt", str(run_a), str(artifact_a), str(run_b), str(artifact_b)]]
    ) == [
        (run_a, None, "trained on tt", artifact_a),
        (run_b, None, "trained on tt", artifact_b),
    ]


def test_multirun_metric_reference_lines():
    figure = plot_multirun_metric(
        [{"label": "FSQ", "rate": 4.0, "mse": 0.1}],
        "mse",
        "MSE",
        "Rate distortion",
        x_metric="rate",
        vertical_reference=43.0,
        vertical_reference_label="Original particle representation (43 bits)",
        horizontal_reference=0.01,
        horizontal_reference_label="Continuous autoencoder",
    )

    labelled_lines = {line.get_label(): line for line in figure.axes[0].lines}
    assert np.all(
        np.asarray(
            labelled_lines["Original particle representation (43 bits)"].get_xdata()
        )
        == 43
    )
    assert np.all(
        np.asarray(labelled_lines["Continuous autoencoder"].get_ydata()) == 0.01
    )


def test_multirun_metric_can_suppress_title():
    figure = plot_multirun_metric(
        [
            {"label": "FSQ small", "plot_family": "FSQ", "rate": 2.0, "mse": 0.2},
            {"label": "FSQ large", "plot_family": "FSQ", "rate": 4.0, "mse": 0.1},
        ],
        "mse",
        "MSE",
        "Rate distortion",
        x_metric="rate",
        show_title=False,
    )

    assert figure.axes[0].get_title() == ""
    legend = figure.axes[0].get_legend()
    assert legend.get_texts()[0].get_fontsize() == 12
    assert legend.legend_handles[0].get_marker() == multirun_marker("FSQ")


def test_multirun_metric_uses_explicit_presentation_marker_areas():
    figure = plot_multirun_metric(
        [
            {
                "label": "small",
                "plot_family": "VQ STE",
                "rate": 0.2,
                "mse": 0.2,
                "plot_marker_area": 30.0,
            },
            {
                "label": "large",
                "plot_family": "VQ STE",
                "rate": 0.3,
                "mse": 0.1,
                "plot_marker_area": 110.0,
            },
        ],
        "mse",
        "MSE",
        "Presentation plot",
        x_metric="rate",
    )

    assert figure.axes[0].collections[0].get_sizes().tolist() == [30.0, 110.0]


def test_multirun_palette_is_stable_across_input_orders():
    records = [
        {"label": "model B", "plot_family": "VQ STE", "rate": 2.0, "mse": 0.2},
        {"label": "model A", "plot_family": "FSQ", "rate": 1.0, "mse": 0.1},
    ]
    forward = plot_multirun_metric(
        records, "mse", "MSE", "Comparison", x_metric="rate"
    )
    reverse = plot_multirun_metric(
        list(reversed(records)), "mse", "MSE", "Comparison", x_metric="rate"
    )

    def colors_by_label(figure):
        return {
            collection.get_label(): tuple(collection.get_facecolor()[0])
            for collection in figure.axes[0].collections
        }

    assert colors_by_label(forward) == colors_by_label(reverse)
    assert multirun_color("FSQ") != multirun_color("VQ STE")
    assert multirun_color("arbitrary run") == multirun_color("arbitrary run")


def test_split_quantizer_families_have_distinct_styles():
    families = (
        "VQ μ + FSQ α=128",
        "VQ μ + FSQ α=64",
        "VQ μ + FSQ α=32",
        "FSQ μ + α=128",
        "FSQ μ + α=64",
        "FSQ μ + α=32",
    )

    assert len({multirun_color(family) for family in families}) == len(families)
    assert len({multirun_marker(family) for family in families}) == len(families)


def test_training_domain_style_aliases_match_presentation_labels():
    tt_labels = ("TT_only", "Trained on tt", "tt-trained VQ-STE")
    mixture_labels = (
        "SM_mixture",
        "Trained on SM mixture",
        "mixture-trained VQ-STE",
    )

    assert len({multirun_color(label) for label in tt_labels}) == 1
    assert len({multirun_marker(label) for label in tt_labels}) == 1
    assert len({multirun_color(label) for label in mixture_labels}) == 1
    assert len({multirun_marker(label) for label in mixture_labels}) == 1


def test_phaedra_split_architectures_are_prefixed_and_stacked():
    families = [
        "VQ STE",
        "FSQ μ + α=64",
        "FAISS k-means",
        "VQ μ + FSQ α=32",
    ]
    ordered = sorted(families, key=multirun_legend_sort_key)

    assert ordered[:2] == ["VQ μ + FSQ α=32", "FSQ μ + α=64"]
    assert multirun_display_label(ordered[0]) == "(PHAEDRA) VQ μ + FSQ α=32"
    assert multirun_display_label(ordered[1]) == "(PHAEDRA) FSQ μ + α=64"

    figure = plot_multirun_metric(
        [
            {"label": family, "plot_family": family, "rate": index + 1, "mse": 0.1}
            for index, family in enumerate(families)
        ],
        "mse",
        "MSE",
        "Comparison",
        x_metric="rate",
    )
    legend_labels = [text.get_text() for text in figure.axes[0].legend().get_texts()]
    assert legend_labels[:2] == [
        "(PHAEDRA) VQ μ + FSQ α=32",
        "(PHAEDRA) FSQ μ + α=64",
    ]


def test_multirun_wandb_upload_uses_detected_credentials(monkeypatch, tmp_path):
    args = SimpleNamespace(
        no_wandb=False,
        wandb_project=None,
        wandb_name=None,
        wandb_group=None,
        wandb_entity=None,
    )
    records = [{"wandb_project": "orbit-tokenizer", "run_dir": tmp_path / "run"}]
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _wandb_upload_settings(args, records, tmp_path / "comparison") is None

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    settings = _wandb_upload_settings(args, records, tmp_path / "comparison")

    assert settings == {
        "project": "orbit-tokenizer",
        "name": "orbit-multirun-comparison",
        "group": "orbit-multirun-comparisons",
        "entity": "test-entity",
    }
