"""Tests for optimizer-step-clocked VQ dead-code replacement."""

import copy

import torch

from vqtorch.nn import VectorQuant
from vqtorch.nn.utils import step_lru_replacement


def _quantizer(timeout: int = 3) -> VectorQuant:
    quantizer = VectorQuant(
        feature_size=1,
        num_codes=4,
        dim=-1,
        kmeans_init=False,
        replace_freq=timeout,
        replace_clock="optimizer",
    )
    with torch.no_grad():
        quantizer.codebook.weight[:, 0] = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    return quantizer.train()


def test_microbatches_do_not_advance_replacement_clock():
    quantizer = _quantizer(timeout=3)
    initial_counts = quantizer._counts.clone()
    values = torch.full((2, 5, 1), -3.0)

    for _ in range(8):
        quantizer(values)

    torch.testing.assert_close(quantizer._counts, initial_counts)
    step_lru_replacement(quantizer)
    torch.testing.assert_close(
        quantizer._counts.sort().values, torch.tensor([2.0, 2.0, 2.0, 3.0])
    )


def test_accumulated_and_physical_batches_have_same_replacement_state():
    physical = _quantizer(timeout=3)
    accumulated = copy.deepcopy(physical)
    values = torch.tensor([[[-3.0]], [[-1.0]], [[-3.0]], [[-1.0]]])

    physical(values)
    for microbatch in values.chunk(4):
        accumulated(microbatch)
    step_lru_replacement(physical)
    step_lru_replacement(accumulated)

    torch.testing.assert_close(physical._counts, accumulated._counts)


def test_replacement_occurs_only_after_timeout_optimizer_steps():
    quantizer = _quantizer(timeout=2)
    values = torch.full((2, 5, 1), -3.0)
    initial_codebook = quantizer.codebook.weight.detach().clone()

    quantizer(values)
    step_lru_replacement(quantizer)
    torch.testing.assert_close(quantizer.codebook.weight, initial_codebook)

    quantizer(values)
    torch.testing.assert_close(quantizer.codebook.weight, initial_codebook)
    step_lru_replacement(quantizer)

    torch.testing.assert_close(quantizer.codebook.weight[0], initial_codebook[0])
    assert not torch.equal(quantizer.codebook.weight[1:], initial_codebook[1:])
    torch.testing.assert_close(quantizer._counts, torch.full((4,), 2.0))


def test_forward_clock_remains_the_legacy_default():
    quantizer = VectorQuant(
        feature_size=1,
        num_codes=4,
        dim=-1,
        kmeans_init=False,
        replace_freq=3,
    ).train()
    values = torch.full((2, 5, 1), -3.0)

    quantizer(values)

    torch.testing.assert_close(
        quantizer._counts.sort().values, torch.tensor([2.0, 2.0, 2.0, 3.0])
    )
