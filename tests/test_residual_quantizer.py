"""Tests for configurable residual vector quantization."""

from functools import partial
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from gabbro.data.data_tokenization import _extract_serialized_tokens, _token_range
from gabbro.models.quantizers import ResidualVectorQuantizer, build_quantizer
from gabbro.models.vqvae import VQVAELightning, VQVAEMLP
from gabbro.plotting.orbit import code_entropy_metrics
from scripts.collect_orbit_multirun import _quantizer_metadata
from scripts.export_orbit_event_binaries import _quantizer_shape


def test_residual_quantizer_selects_and_decodes_each_stage():
    quantizer = ResidualVectorQuantizer(
        feature_size=1,
        num_quantizers=2,
        codebook_size=2,
    )
    quantizer.eval()
    with torch.no_grad():
        quantizer.codebooks.copy_(torch.tensor([[[0.0], [2.0]], [[0.0], [1.0]]]))

    z = torch.tensor([[2.8]])
    z_q, output = quantizer(z)

    assert output["q"].tolist() == [[1, 1]]
    assert torch.allclose(z_q, torch.tensor([[3.0]]))
    assert torch.allclose(quantizer.decode_tokens(output["q"]), torch.tensor([[3.0]]))
    assert output["stage_loss"].shape == (1, 2)


def test_residual_quantizer_paper_aligned_defaults_and_checkpoint_round_trip():
    quantizer = ResidualVectorQuantizer(
        feature_size=2,
        num_quantizers=2,
        codebook_size=3,
    )
    assert quantizer.commitment_weight == 1.0
    assert quantizer.ema_decay == 0.8
    assert torch.count_nonzero(quantizer.ema_cluster_size) == 0

    tokens = torch.tensor([[[0, 1], [2, 0]]])
    expected = quantizer.decode_tokens(tokens)
    restored = ResidualVectorQuantizer(
        feature_size=2,
        num_quantizers=2,
        codebook_size=3,
    )
    restored.load_state_dict(quantizer.state_dict())
    assert torch.equal(restored.decode_tokens(tokens), expected)


def test_residual_quantizer_uses_straight_through_gradient_and_ema_updates():
    torch.manual_seed(3)
    quantizer = ResidualVectorQuantizer(
        feature_size=2,
        num_quantizers=3,
        codebook_size=4,
        commitment_weight=0.5,
        ema_decay=0.9,
    )
    quantizer.train()
    original_codebooks = quantizer.codebooks.clone()
    z = torch.randn(8, 2, requires_grad=True)

    z_q, output = quantizer(z)
    loss = z_q.square().mean() + output["loss_weighted"]
    loss.backward()

    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))
    assert not torch.equal(quantizer.codebooks, original_codebooks)
    assert output["q"].shape == (8, 3)

    codebooks_after_training = quantizer.codebooks.clone()
    with torch.no_grad():
        quantizer(torch.randn(8, 2))
    assert torch.equal(quantizer.codebooks, codebooks_after_training)


def test_residual_quantizer_factory_is_mutually_exclusive_with_split_quantization():
    quantizer = build_quantizer(
        feature_size=4,
        vq_kwargs={"num_codes": 8},
        residual_quantizer_cfg={"num_quantizers": 2, "codebook_size": 4},
    )
    assert isinstance(quantizer, ResidualVectorQuantizer)

    with pytest.raises(ValueError, match="cannot be enabled together"):
        build_quantizer(
            feature_size=4,
            vq_kwargs={"num_codes": 8},
            split_quantizer_cfg={"mode": "split"},
            residual_quantizer_cfg={"num_quantizers": 2, "codebook_size": 4},
        )


def test_residual_entropy_sums_stage_rates():
    codes = np.array([[[0, 0], [1, 1], [0, 1]]])
    mask = np.array([[True, True, False]])
    metrics = code_entropy_metrics(
        codes,
        mask,
        mask,
        num_codes=4,
        num_quantizers=2,
    )

    assert metrics["metrics/entropy/stage_0/bits_per_token"] == pytest.approx(1.0)
    assert metrics["metrics/entropy/stage_1/bits_per_token"] == pytest.approx(1.0)
    assert metrics["metrics/entropy/marginal_bits_per_token"] == pytest.approx(2.0)
    assert metrics["metrics/entropy/fixed_bits_per_token"] == 4
    assert metrics["metrics/rate/fixed_bits_per_input_particle"] == pytest.approx(4.0)
    assert metrics["metrics/total_codebook_size"] == 16
    assert metrics["metrics/codebook/stored_vectors"] == 8


def test_residual_entropy_uses_fixed_width_for_non_power_of_two_codebooks():
    codes = np.array([[[0, 0], [1, 2], [2, 1]]])
    mask = np.ones((1, 3), dtype=bool)
    metrics = code_entropy_metrics(
        codes,
        mask,
        mask,
        num_codes=3,
        num_quantizers=2,
    )

    assert metrics["metrics/entropy/fixed_bits_per_token"] == 4
    assert metrics["metrics/total_codebook_size"] == 9
    assert metrics["metrics/codebook/stored_vectors"] == 6


def test_residual_tokens_survive_parquet_and_restore_stage_order(tmp_path):
    quantizer = ResidualVectorQuantizer(
        feature_size=2,
        num_quantizers=2,
        codebook_size=4,
    )
    model = SimpleNamespace(model=SimpleNamespace(vqlayer=quantizer))
    stage_tokens = ak.Array(
        {
            "part_token_stage_1": [[2, 3], [1]],
            "part_token_stage_0": [[0, 1], [3]],
        }
    )
    wrapped = ak.Array({"part_token_id": stage_tokens})
    path = tmp_path / "rvq_tokens.parquet"
    ak.to_parquet(wrapped, path)

    restored = _extract_serialized_tokens(ak.from_parquet(path), model)

    assert restored.fields == ["part_token_stage_0", "part_token_stage_1"]
    assert ak.to_list(restored.part_token_stage_0) == [[0, 1], [3]]
    assert ak.to_list(restored.part_token_stage_1) == [[2, 3], [1]]
    assert _token_range(restored) == (0, 3)


def test_residual_awkward_tokens_decode_through_mlp():
    module = VQVAELightning(
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        model_type="MLP",
        model_kwargs={
            "input_dim": 2,
            "latent_dim": 2,
            "vq_kwargs": {},
            "residual_quantizer_cfg": {
                "num_quantizers": 2,
                "codebook_size": 4,
            },
        },
    )
    module.eval()
    with torch.no_grad():
        module.model.vqlayer.codebooks.copy_(
            torch.tensor(
                [
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
                ]
            )
        )
    tokens = ak.Array(
        {
            "part_token_stage_1": [[1, 2], [3]],
            "part_token_stage_0": [[1, 2], [3]],
        }
    )

    reconstructed = module.reconstruct_ak_tokens(
        tokens,
        pp_dict={"feature_0": None, "feature_1": None},
        pad_length=2,
        hide_pbar=True,
    )

    assert reconstructed.fields == ["feature_0", "feature_1"]
    assert ak.num(reconstructed.feature_0, axis=1).tolist() == [2, 1]


def test_residual_quantizer_integrates_with_masked_mlp():
    model = VQVAEMLP(
        input_dim=4,
        latent_dim=3,
        vq_kwargs={},
        residual_quantizer_cfg={"num_quantizers": 2, "codebook_size": 4},
    )
    comparison_model = VQVAEMLP(
        input_dim=4,
        latent_dim=3,
        vq_kwargs={},
        residual_quantizer_cfg={"num_quantizers": 2, "codebook_size": 4},
    )
    comparison_model.load_state_dict(model.state_dict())
    model.train()
    comparison_model.train()
    mask = torch.tensor([[True, False, True], [True, True, False]])
    inputs = torch.randn(2, 3, 4)
    changed_padding = inputs.clone()
    changed_padding[~mask] = 1e6
    reconstructed, output = model(inputs, mask=mask)
    _, comparison_output = comparison_model(changed_padding, mask=mask)

    assert reconstructed.shape == (2, 3, 4)
    assert output["q"].shape == (2, 3, 2)
    assert torch.count_nonzero(output["q"][~mask]) == 0
    assert torch.count_nonzero(output["z_q"][~mask]) == 0
    assert torch.equal(output["q"], comparison_output["q"])
    assert torch.equal(
        model.vqlayer.codebooks,
        comparison_model.vqlayer.codebooks,
    )


def test_residual_quantizer_integrates_with_masked_transformer_model_step():
    module = VQVAELightning(
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        model_type="VQVAETransformer",
        model_kwargs={
            "input_dim": 4,
            "hidden_dim": 8,
            "latent_dim": 4,
            "num_heads": 2,
            "num_blocks": 1,
            "alpha": 10.0,
            "vq_kwargs": {},
            "residual_quantizer_cfg": {
                "num_quantizers": 3,
                "codebook_size": 4,
                "commitment_weight": 1.0,
            },
        },
    )
    batch = {
        "part_features": torch.randn(2, 3, 4),
        "part_mask": torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        ),
        "jet_type_labels": torch.zeros(2, dtype=torch.long),
    }

    result = module.model_step(batch, return_x=True)
    loss, metrics, _, _, _, _, code_idx, code_mask, *_ = result

    assert code_idx.shape == (2, 3, 3)
    assert code_mask.shape == (2, 3)
    assert torch.all((code_idx[code_mask] >= 0) & (code_idx[code_mask] < 4))
    assert torch.isfinite(loss)
    assert metrics["loss_quantizer"] > 0
    assert metrics["loss_quantizer_weighted"] > 0
    assert torch.equal(
        metrics["loss_quantizer_weighted"], metrics["loss_quantizer"]
    )
    assert "loss_quantizer_stage_2" in metrics


def test_residual_quantizer_metadata_describes_depth_and_per_stage_size():
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "gabbro.models.vqvae.VQVAELightning",
                "model_kwargs": {
                    "quantization_enabled": True,
                    "vq_kwargs": {"num_codes": 4096},
                    "residual_quantizer_cfg": {
                        "num_quantizers": 3,
                        "codebook_size": 16,
                    },
                },
            }
        }
    )

    metadata = _quantizer_metadata(cfg)
    assert metadata["quantizer_mode"] == "residual_vq"
    assert metadata["num_quantizers"] == 3
    assert metadata["per_stage_codebook_size"] == 16
    assert metadata["total_codebook_size"] == 4096
    assert metadata["stored_code_vectors"] == 48
    assert metadata["fixed_bits_per_latent"] == 12
    assert _quantizer_shape(cfg) == (16, 3)
