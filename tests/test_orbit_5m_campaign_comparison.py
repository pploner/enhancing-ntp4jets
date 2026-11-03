"""Focused tests for the restartable joint 5M campaign plot workflow."""

import json

import numpy as np
import pytest
import yaml

pytest.importorskip("mplhep")

from gabbro.plotting.writeup.model_comparison import (
    load_pid_confusion,
    load_registry,
    plot_downstream,
    plot_jets,
    select_downstream_models,
)
from scripts.evaluate_orbit_5m_plot_cache import load_exported_config, split_quota


def test_split_quota_preserves_balanced_event_budget():
    assert split_quota(10_000, 3) == [3334, 3333, 3333]
    assert sum(split_quota(2000, 9)) == 2000


def test_wandb_config_unwraps_nested_hydra_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"value": "{'sequence_type': 'particle_full', 'max_sequence_length': 500}"},
                "seed": {"value": 12345},
            }
        )
    )
    config = load_exported_config(path)
    assert config.data.sequence_type == "particle_full"
    assert config.data.max_sequence_length == 500
    assert config.seed == 12345


def test_registry_assigns_eight_unique_styles(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": [{"id": str(i)} for i in range(8)]}))
    models = load_registry(path)
    assert len({model["color"] for model in models}) == 8
    assert len({model["marker"] for model in models}) == 8


def test_pid_confusion_loader_recomputes_row_normalization(tmp_path):
    source = tmp_path / "ttbar" / "artifacts"
    source.mkdir(parents=True)
    counts = np.eye(8, dtype=np.int64)
    counts[6, 6] = 3
    counts[6, 7] = 1
    path = source / "pid_confusion_matrix_test_suite_0.json"
    path.write_text(json.dumps({"counts": counts.tolist()}))

    loaded_path, loaded_counts, normalized, payload = load_pid_confusion(
        {"pid_confusion_dir": str(tmp_path)}, "ttbar"
    )

    assert loaded_path == path
    assert np.array_equal(loaded_counts, counts)
    assert payload["counts"] == counts.tolist()
    assert normalized[6, 6] == pytest.approx(0.75)
    assert normalized[6, 7] == pytest.approx(0.25)
    assert np.allclose(normalized.sum(axis=1), 1.0)


def test_jet_plot_accepts_unconstrained_hungarian_cache(tmp_path):
    models = []
    suite_dir = tmp_path / "ttbar"
    suite_dir.mkdir()
    for index in range(8):
        model = {
            "id": f"m{index}",
            "label": f"model {index}",
            "color": f"C{index}",
            "linestyle": "-",
        }
        models.append(model)
        path = suite_dir / f"m{index}.npz"
        np.savez_compressed(
            path,
            jet_delta_pt=np.linspace(-0.2, 0.2, 20),
            jet_delta_eta=np.linspace(-0.1, 0.1, 20),
            jet_delta_phi=np.linspace(-0.1, 0.1, 20),
            jet_match_dr=np.linspace(0.0, 2.0, 20),
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "jet_pairing": (
                        "Hungarian minimum total delta-R, no delta-R rejection"
                    )
                }
            )
        )
    figure = plot_jets(tmp_path, "ttbar", models)
    figure.canvas.draw()
    assert figure._suptitle is None
    assert len(figure.axes) == 3
    assert all(axis.get_yscale() == "log" for axis in figure.axes)
    renderer = figure.canvas.get_renderer()
    legend_bottom = figure.legends[0].get_window_extent(renderer).y0
    axes_top = max(axis.get_window_extent(renderer).y1 for axis in figure.axes)
    assert legend_bottom >= axes_top
    identity = " ".join(text.get_text() for text in figure.texts)
    assert "Test: $t\\bar{t}$, balanced decay modes (10k)" in identity
    assert "AK4 anti-$k_t$" in identity
    assert "Hungarian pairing" in identity


def test_paper_downstream_selection_omits_only_pid_weighted_models():
    models = [
        {"id": "tt", "selection": "full500"},
        {"id": "sm", "selection": "full500"},
        {"id": "pid", "selection": "full500", "pid_weighted": True},
        {"id": "filtered", "selection": "puppi128"},
    ]

    assert [
        model["id"]
        for model in select_downstream_models(models, "full500", paper=False)
    ] == ["tt", "sm", "pid"]
    assert [
        model["id"]
        for model in select_downstream_models(models, "full500", paper=True)
    ] == ["tt", "sm"]


def test_downstream_ratio_rejects_mismatched_original_samples(tmp_path):
    suite_dir = tmp_path / "gghbb"
    suite_dir.mkdir()
    models = []
    for index in range(2):
        model = {
            "id": f"m{index}",
            "label": f"model {index}",
            "color": f"C{index}",
            "linestyle": "-",
        }
        models.append(model)
        path = suite_dir / f"m{index}.npz"
        np.savez_compressed(
            path,
            higgs_n_jets_original=np.array([2.0, 3.0]),
            higgs_n_jets_decoded=np.array([2.0, 3.0]),
            higgs_mass_original=np.array([100.0 + index, 125.0]),
            higgs_mass_decoded=np.array([100.0, 125.0]),
        )
        path.with_suffix(".json").write_text("{}")

    with pytest.raises(ValueError, match="shared tokenized/original denominator"):
        plot_downstream(tmp_path, "full500", models, "higgs")


def test_tau32_uses_model_specific_matched_original_denominators(tmp_path):
    suite_dir = tmp_path / "tau32"
    suite_dir.mkdir()
    models = []
    for index in range(2):
        model = {
            "id": f"m{index}",
            "label": f"model {index}",
            "color": f"C{index}",
            "linestyle": "-",
        }
        models.append(model)
        path = suite_dir / f"m{index}.npz"
        np.savez_compressed(
            path,
            original_count=np.array([1, 2]),
            decoded_count=np.array([1, 2]),
            original_tau32=np.array([0.5 + 0.1 * index, 0.7]),
            decoded_tau32=np.array([0.55, 0.75]),
        )
        path.with_suffix(".json").write_text("{}")

    figure = plot_downstream(tmp_path, "full500", models, "tau32")
    labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert labels[0] == "Original (matched samples)"
    assert len(figure.axes) == 5
