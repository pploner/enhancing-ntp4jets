"""Tests for auxiliary reconstructed-jet tokens in the ORBIT VQ-VAE."""

from functools import partial

import torch

from gabbro.models.vqvae import VQVAELightning


def _jet_token_module(
    jet_loss_weight: float,
    *,
    quantization_enabled: bool = False,
) -> VQVAELightning:
    vq_kwargs = (
        {"num_codes": 8, "replace_freq": 0, "rotation_tr": False}
        if quantization_enabled
        else {}
    )
    return VQVAELightning(
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        model_type="VQVAETransformer",
        model_kwargs={
            "input_dim": 4,
            "hidden_dim": 8,
            "latent_dim": 4,
            "num_heads": 2,
            "num_blocks": 1,
            "alpha": 1.0,
            "quantization_enabled": quantization_enabled,
            "vq_kwargs": vq_kwargs,
            "token_type_count": 2,
            "jet_reconstruction_loss_weight": jet_loss_weight,
            "pid_cfg": {
                "enabled": True,
                "num_classes": 8,
                "loss_weight": 1.0,
            },
        },
    )


def _jet_token_batch() -> dict[str, torch.Tensor]:
    return {
        "part_features": torch.randn(2, 3, 4),
        "part_mask": torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        ),
        "part_pid": torch.tensor([[1, 3, -1], [6, 0, 2]]),
        "jet_features": torch.randn(2, 2, 4),
        "jet_mask": torch.tensor([[True, False], [True, True]], dtype=torch.bool),
        "jet_type_labels": torch.zeros(2, dtype=torch.long),
    }


def test_jet_tokens_extend_quantized_sequence_but_not_particle_output():
    module = _jet_token_module(jet_loss_weight=0.0)
    module.eval()
    batch = _jet_token_batch()

    result = module.model_step(batch, return_x=True)
    _, metrics, _, particle_reco, _, _, code_idx, code_mask, jets, jet_reco, jet_mask = result

    assert particle_reco.shape == (2, 3, 4)
    assert jets.shape == jet_reco.shape == (2, 2, 4)
    assert torch.equal(jet_mask, batch["jet_mask"])
    assert code_idx.shape == code_mask.shape == (2, 5)
    assert torch.equal(
        code_mask,
        torch.cat([batch["part_mask"], batch["jet_mask"]], dim=1),
    )
    assert torch.allclose(metrics["mean_valid_jets"], torch.tensor(1.5))
    assert metrics["loss_reco_jet"] > 0
    assert metrics["loss_reco_jet_weighted"] == 0


def test_unit_jet_loss_adds_separately_normalized_jet_objective():
    torch.manual_seed(7)
    no_loss = _jet_token_module(jet_loss_weight=0.0)
    with_loss = _jet_token_module(jet_loss_weight=1.0)
    with_loss.load_state_dict(no_loss.state_dict())
    no_loss.eval()
    with_loss.eval()
    batch = _jet_token_batch()

    loss_without, metrics_without = no_loss.model_step(batch)
    loss_with, metrics_with = with_loss.model_step(batch)

    assert torch.allclose(metrics_without["loss_reco_particle"], metrics_with["loss_reco_particle"])
    assert torch.allclose(metrics_without["loss_reco_jet"], metrics_with["loss_reco_jet"])
    assert torch.allclose(
        loss_with - loss_without,
        metrics_with["loss_reco_jet"],
        atol=1e-6,
    )
    assert torch.allclose(
        metrics_with["loss_reco_combined"],
        metrics_with["loss_reco_particle"] + metrics_with["loss_reco_jet"],
    )


def test_vq_ste_quantizes_all_valid_particle_and_jet_tokens():
    module = _jet_token_module(
        jet_loss_weight=0.0,
        quantization_enabled=True,
    )
    batch = _jet_token_batch()

    result = module.model_step(batch, return_x=True)
    loss, metrics, _, _, _, _, code_idx, code_mask, *_ = result

    expected_mask = torch.cat([batch["part_mask"], batch["jet_mask"]], dim=1)
    assert code_idx.shape == expected_mask.shape
    assert torch.equal(code_mask, expected_mask)
    assert torch.all((code_idx[code_mask] >= 0) & (code_idx[code_mask] < 8))
    assert torch.isfinite(loss)
    assert metrics["loss_quantizer"] > 0
