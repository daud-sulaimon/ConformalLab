"""Tests for the pure helper functions in run.py (no network/model calls)."""

import numpy as np

from run import _softmax, _top_k_accuracy


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = _softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_top_k_accuracy_perfect_predictions():
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    labels = np.array([0, 1])
    assert _top_k_accuracy(probs, labels, k=1) == 1.0


def test_top_k_accuracy_zero_when_wrong():
    probs = np.array([[0.9, 0.05, 0.05]])
    labels = np.array([1])
    assert _top_k_accuracy(probs, labels, k=1) == 0.0