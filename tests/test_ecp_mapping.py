# tests/test_ecp_mapping.py
"""Verify the ECP threshold mapping derivation (Kasa et al. Eq. 4
adapted to negatively-oriented LAC scores)."""

import numpy as np

from run_ecp import _lac_sets_with_scaled_threshold


def test_scale_of_one_reproduces_unscaled_lac():
    probs = np.array([[0.5, 0.3, 0.15, 0.05]])
    q_hat = 0.9
    sets = _lac_sets_with_scaled_threshold(probs, q_hat, scale=1.0)
    # tau_D = 0.1, threshold = 0.1 -> include p >= 0.1 -> classes 0,1,2
    assert sorted(sets[0]) == [0, 1, 2]


def test_larger_scale_widens_sets_not_narrows():
    probs = np.array([[0.5, 0.3, 0.15, 0.05]])
    q_hat = 0.9
    small = _lac_sets_with_scaled_threshold(probs, q_hat, scale=1.0)
    large = _lac_sets_with_scaled_threshold(probs, q_hat, scale=3.0)
    # scale=3 -> threshold 0.1/3 = 0.033 -> now includes class 3 too
    assert len(large[0]) >= len(small[0])
    assert sorted(large[0]) == [0, 1, 2, 3]


def test_sets_are_never_empty():
    probs = np.array([[0.9, 0.05, 0.03, 0.02]])
    sets = _lac_sets_with_scaled_threshold(probs, q_hat=0.999, scale=1.0)
    assert len(sets[0]) >= 1