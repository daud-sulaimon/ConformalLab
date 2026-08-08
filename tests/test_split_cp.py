"""Tests for src.conformal.split_cp."""

import numpy as np
import pytest

from src.conformal.split_cp import SplitConformalMethod


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match="alpha"):
        SplitConformalMethod(alpha=1.5)


def test_predict_sets_before_calibrate_raises():
    method = SplitConformalMethod(alpha=0.1)
    with pytest.raises(RuntimeError, match="calibrate"):
        method.predict_sets(np.random.rand(5, 3))


def test_predict_sets_returns_one_list_per_sample():
    rng = np.random.default_rng(42)
    calibration_probs = rng.dirichlet(np.ones(5), size=200)
    calibration_labels = rng.integers(0, 5, size=200)

    method = SplitConformalMethod(alpha=0.1)
    method.calibrate(calibration_probs, calibration_labels)

    test_probs = rng.dirichlet(np.ones(5), size=50)
    sets = method.predict_sets(test_probs)

    assert len(sets) == 50
    assert all(isinstance(s, list) for s in sets)


def test_empirical_coverage_approximately_matches_target():
    """
    The core correctness check: on data where the model's probabilities
    are genuinely informative, empirical test-set coverage should be
    close to (at least) 1 - alpha, verifying the calibration formula
    is implemented correctly rather than just "running without errors."
    """
    rng = np.random.default_rng(0)
    num_classes = 10
    alpha = 0.1

    def make_probs_and_labels(n):
        labels = rng.integers(0, num_classes, size=n)
        # Simulate a model that's reasonably good: true class gets a
        # boosted, noisy score, others get small random scores.
        probs = rng.uniform(0.01, 0.2, size=(n, num_classes))
        probs[np.arange(n), labels] += rng.uniform(0.3, 0.8, size=n)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs, labels

    calibration_probs, calibration_labels = make_probs_and_labels(2000)
    test_probs, test_labels = make_probs_and_labels(2000)

    method = SplitConformalMethod(alpha=alpha)
    method.calibrate(calibration_probs, calibration_labels)
    sets = method.predict_sets(test_probs)

    covered = [
        label in prediction_set
        for label, prediction_set in zip(test_labels, sets)
    ]
    empirical_coverage = np.mean(covered)

    # Should be close to 0.9, and essentially never far below it given
    # a large sample size (allowing small statistical slack).
    assert empirical_coverage >= (1 - alpha) - 0.03