"""Tests for src.conformal.nonconformity."""

import numpy as np
import pytest

from src.conformal.nonconformity import (
    aps_scores,
    conformal_quantile,
    lac_scores,
    raps_scores,
)


def test_lac_scores_known_values():
    probs = np.array([[0.9, 0.1], [0.3, 0.7]])
    labels = np.array([0, 1])
    scores = lac_scores(probs, labels)
    assert scores == pytest.approx([0.1, 0.3])


def test_aps_scores_known_values():
    probs = np.array([[0.5, 0.3, 0.2]])
    labels = np.array([0])
    scores = aps_scores(probs, labels)
    assert scores == pytest.approx([0.5])

    labels2 = np.array([1])
    scores2 = aps_scores(probs, labels2)
    assert scores2 == pytest.approx([0.8])


def test_raps_scores_penalizes_low_rank():
    probs = np.array([[0.5, 0.3, 0.2]])
    labels = np.array([2])
    scores = raps_scores(probs, labels, lam=0.1, k_reg=1)
    assert scores == pytest.approx([1.2])


def test_conformal_quantile_matches_manual_calculation():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    q_hat = conformal_quantile(scores, alpha=0.4)
    assert q_hat == pytest.approx(0.5)