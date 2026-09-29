"""Tests for finite scalar quantizer mappings and token round-trips."""

from __future__ import annotations

import pytest
import torch

from gabbro.models.quantizers import FSQ, SplitQuantizer


@pytest.mark.parametrize("level", [2, 3, 4, 8, 64])
def test_reference_fsq_reaches_exactly_declared_levels(level: int):
    quantizer = FSQ([level], mapping="reference")
    inputs = torch.linspace(-20, 20, 200_001).unsqueeze(-1)

    z_quantized, codes, _ = quantizer(inputs)

    assert torch.unique(codes).numel() == level
    assert torch.unique(z_quantized).numel() == level
    assert codes.min().item() == 0
    assert codes.max().item() == level - 1


@pytest.mark.parametrize("levels", [[2], [4], [8], [64], [5, 4, 3, 2], [1, 128]])
def test_reference_fsq_forward_tokens_decode_to_same_latent(levels: list[int]):
    torch.manual_seed(7)
    quantizer = FSQ(levels, mapping="reference")
    inputs = torch.randn(4096, len(levels)) * 5

    z_quantized, codes, _ = quantizer(inputs)
    z_from_tokens = quantizer.decode_tokens(codes)

    torch.testing.assert_close(z_from_tokens, z_quantized, rtol=0, atol=1e-7)
    assert torch.all(codes >= 0)
    assert torch.all(codes < quantizer.num_codes)


def test_reference_fsq_decodes_every_mixed_radix_token():
    quantizer = FSQ([2, 3, 4], mapping="reference")
    tokens = torch.arange(quantizer.num_codes)

    decoded = quantizer.decode_tokens(tokens)

    assert decoded.shape == (24, 3)
    assert torch.unique(decoded, dim=0).shape[0] == quantizer.num_codes


def test_reference_fsq_level_one_is_fixed_zero_dimension():
    quantizer = FSQ([1], mapping="reference")
    inputs = torch.tensor([[-100.0], [0.0], [100.0]], requires_grad=True)

    z_quantized, codes, loss = quantizer(inputs)
    z_quantized.sum().backward()

    torch.testing.assert_close(z_quantized, torch.zeros_like(z_quantized))
    torch.testing.assert_close(codes, torch.zeros_like(codes))
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs))


def test_legacy_fsq_preserves_even_level_mapping_and_decode_mismatch():
    quantizer = FSQ([4], mapping="legacy")
    inputs = torch.tensor([[-1.0], [-0.2], [0.2], [1.0]])

    z_quantized, codes, _ = quantizer(inputs)

    torch.testing.assert_close(
        z_quantized.squeeze(-1), torch.tensor([-2 / 3, 0.0, 0.0, 2 / 3])
    )
    torch.testing.assert_close(codes, torch.tensor([0, 1, 1, 2]))
    assert not torch.equal(quantizer.decode_tokens(codes), z_quantized)


def _split_cfg(*, mapping: str | None) -> dict:
    cfg = {
        "mode": "split",
        "branch_order": ["mu"],
        "quantizer": "fsq",
        "fsq_mu_levels": [2, 3, 4],
        "bypass_single_branch_projection": True,
    }
    if mapping is not None:
        cfg["fsq_mapping"] = mapping
    return cfg


def test_split_quantizer_reference_tokens_round_trip():
    quantizer = SplitQuantizer(feature_size=3, cfg=_split_cfg(mapping="reference"))
    inputs = torch.randn(8, 11, 3)
    mask = torch.ones(8, 11)

    z_quantized, output = quantizer(inputs, mask=mask)
    z_from_tokens = quantizer.decode_tokens(output["branch_q"], mask=mask)

    torch.testing.assert_close(z_from_tokens, z_quantized, rtol=0, atol=1e-7)


def test_split_quantizer_missing_mapping_uses_legacy_compatibility():
    quantizer = SplitQuantizer(feature_size=3, cfg=_split_cfg(mapping=None))

    assert quantizer.quantizers["mu"].mapping == "legacy"


def test_fsq_rejects_invalid_mapping_and_levels():
    with pytest.raises(ValueError, match="positive integers"):
        FSQ([2, 0])
    with pytest.raises(ValueError, match="either 'reference' or 'legacy'"):
        FSQ([2], mapping="unknown")
