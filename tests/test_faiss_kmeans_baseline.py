"""Tests for the optional GPU FAISS particle baseline."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("awkward")
pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

from gabbro.models.vqvae import FaissKMeansBaselineLightning


class _FakeIndex:
    def __init__(self, centroids):
        self.centroids = centroids

    def search(self, values, count):
        assert count == 1
        distances = np.sum((values[:, None] - self.centroids[None, :]) ** 2, axis=-1)
        indices = np.argmin(distances, axis=1)[:, None]
        selected = np.take_along_axis(distances, indices, axis=1)
        return selected.astype(np.float32), indices.astype(np.int64)


class _FakeKmeans:
    def __init__(self, d, k, **kwargs):
        self.d = d
        self.k = k
        self.kwargs = kwargs
        self.obj = []

    def train(self, values):
        assert values.shape[1] == self.d
        self.centroids = np.ascontiguousarray(values[: self.k])
        self.index = _FakeIndex(self.centroids)
        self.obj = [3.5]


class _FakeFlatIndex(_FakeIndex):
    def __init__(self, dimension):
        self.dimension = dimension
        self.centroids = None

    def add(self, centroids):
        assert centroids.shape[1] == self.dimension
        self.centroids = centroids


def _fake_faiss_module():
    return SimpleNamespace(
        __version__="test",
        get_num_gpus=lambda: 1,
        Kmeans=_FakeKmeans,
        IndexFlatL2=_FakeFlatIndex,
    )


def _datamodule():
    batch = {
        "part_features": torch.tensor(
            [
                [[0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.1, 0.1]],
                [[1.0, 0.0, 1.0, 1.0], [1.1, 0.1, 0.9, 1.1]],
            ],
            dtype=torch.float32,
        ),
        "part_mask": torch.ones((2, 2), dtype=torch.bool),
        "jet_type_labels": torch.tensor([0, 1]),
    }
    return SimpleNamespace(
        hparams=SimpleNamespace(class_to_label={"ggHbb": 0, "minbias": 1}),
        _class_specs={
            "ggHbb": {"weight": 1.0, "max_train_events": 100},
            "minbias": {"weight": 1.0, "max_train_events": 100},
        },
        train_dataloader=lambda: [batch],
    )


def test_class_fit_quotas_follow_training_weights():
    datamodule = _datamodule()
    datamodule._class_specs["ggHbb"]["weight"] = 3.0

    quotas = FaissKMeansBaselineLightning._class_fit_quotas(datamodule, 100)

    assert quotas == {0: ("ggHbb", 75), 1: ("minbias", 25)}


def test_fake_faiss_fit_and_masked_reconstruction(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss_module())
    model = FaissKMeansBaselineLightning(
        num_codes=2,
        fit_max_particles=4,
        fit_max_batches=1,
        save_codebook=False,
    )
    model._trainer = SimpleNamespace(
        datamodule=_datamodule(),
        is_global_zero=True,
        loggers=[],
    )

    model._fit_faiss()
    values = torch.tensor(
        [[[0.01, 0.99, 0.01, 0.01], [9.0, 9.0, 9.0, 9.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, False]])
    reconstructed, code_idx, pid_logits = model._quantize_batch(values, mask)

    assert model._fit_metadata["fit_particles_per_class"] == {"ggHbb": 2, "minbias": 2}
    assert model._fit_metadata["final_objective"] == 3.5
    assert code_idx.tolist() == [[0, 0]]
    assert torch.allclose(reconstructed[0, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.count_nonzero(reconstructed[0, 1]) == 0
    assert pid_logits is None


def test_pid_aware_faiss_uses_joint_one_hot_features(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss_module())
    datamodule = _datamodule()
    datamodule.train_dataloader()[0]["part_pid"] = torch.tensor([[0, 1], [2, 3]])
    model = FaissKMeansBaselineLightning(
        num_codes=2,
        fit_max_particles=4,
        fit_max_batches=1,
        save_codebook=False,
        pid_cfg={"enabled": True, "num_classes": 8, "loss_weight": 1.0},
    )
    model._trainer = SimpleNamespace(
        datamodule=datamodule,
        is_global_zero=True,
        loggers=[],
    )

    model._fit_faiss()
    features = datamodule.train_dataloader()[0]["part_features"][:1]
    mask = torch.ones((1, 2), dtype=torch.bool)
    pid = torch.tensor([[0, 1]])
    reconstructed, code_idx, pid_logits = model._quantize_batch(features, mask, pid)

    assert model.centroids.shape == (2, 12)
    assert reconstructed.shape == features.shape
    assert code_idx.shape == mask.shape
    assert pid_logits.shape == (1, 2, 8)
    assert pid_logits.argmax(dim=-1).tolist() == [[0, 1]]


def test_local_codebook_artifacts(tmp_path):
    model = FaissKMeansBaselineLightning(num_codes=2, fit_max_particles=2)
    model.centroids = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]])
    model._fit_metadata = {"num_codes": 2}
    model._trainer = SimpleNamespace(
        default_root_dir=tmp_path,
        is_global_zero=True,
        loggers=[],
    )

    model._save_codebook_artifacts()

    centroid_file = tmp_path / "artifacts" / "faiss_kmeans_centroids.npz"
    metadata_file = tmp_path / "artifacts" / "faiss_kmeans_metadata.json"
    assert np.load(centroid_file)["centroids"].shape == (2, 4)
    assert json.loads(metadata_file.read_text()) == {"num_codes": 2}


def test_load_saved_codebook_without_training_data(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss_module())
    codebook_path = tmp_path / "faiss_kmeans_centroids.npz"
    centroids = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    np.savez_compressed(codebook_path, centroids=centroids)
    model = FaissKMeansBaselineLightning(
        num_codes=2,
        fit_max_particles=2,
        use_gpu=False,
        codebook_path=codebook_path,
        save_codebook=False,
    )

    model._fit_faiss()
    values = torch.tensor([[[0.01, 0.99, 0.01, 0.01]]], dtype=torch.float32)
    reconstructed, code_idx, _ = model._quantize_batch(
        values,
        torch.tensor([[True]]),
    )

    assert model._fit_metadata["loaded_codebook_path"] == str(codebook_path.resolve())
    assert model._fit_metadata["fit_seconds"] == 0.0
    assert code_idx.tolist() == [[0]]
    assert torch.allclose(reconstructed[0, 0], torch.tensor(centroids[0]))
