"""
Deterministic seeding for ConformalLab.

Ensures that every source of randomness used across the project —
Python's built-in `random`, NumPy, and PyTorch (CPU and CUDA) — is
seeded identically on every run, so that experiments are exactly
reproducible: the same config, run twice, must produce the same
calibration split, the same predictions, and the same metrics.

This module has one responsibility: setting seeds. It does not know
about datasets, models, or experiments.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Seed every random number generator used in ConformalLab.

    Parameters
    ----------
    seed
        Integer seed value applied to Python's `random`, NumPy, and
        PyTorch (CPU and, if available, CUDA). Defaults to 42, matching
        the seed specified in the project's experiment configuration
        convention (see `configs/default.yaml`).

    Notes
    -----
    Also sets `torch.backends.cudnn.deterministic = True` and
    `torch.backends.cudnn.benchmark = False`. This forces cuDNN to use
    deterministic algorithms on GPU, at a small performance cost, which
    is a necessary trade-off for exact reproducibility.

    Examples
    --------
    >>> set_seed(42)
    >>> # every subsequent random operation is now deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PYTHONHASHSEED affects hashing-based randomness (e.g. dict/set
    # iteration order in some edge cases); setting it here only affects
    # child processes, but it is included for completeness and to
    # document the full reproducibility surface.
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed} (Python, NumPy, PyTorch, CUDA).")