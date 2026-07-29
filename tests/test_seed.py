"""Tests for src.utils.seed."""

import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_makes_python_random_reproducible() -> None:
    set_seed(42)
    first = [random.random() for _ in range(5)]

    set_seed(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_set_seed_makes_numpy_reproducible() -> None:
    set_seed(42)
    first = np.random.rand(5)

    set_seed(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_set_seed_makes_torch_reproducible() -> None:
    set_seed(42)
    first = torch.rand(5)

    set_seed(42)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_different_seeds_produce_different_output() -> None:
    set_seed(1)
    first = torch.rand(5)

    set_seed(2)
    second = torch.rand(5)

    assert not torch.equal(first, second)