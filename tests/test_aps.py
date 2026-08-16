"""Tests for src.conformal.aps."""

import numpy as np
import pytest

from src.conformal.aps import APSConformalMethod


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match="alpha"):
        APSConformalMethod(alpha=1.5)


def test_predict_sets_before_calibrate_raises():
    method = APSConformalMethod(alpha=0.1)
    with pytest.raises(RuntimeError, match="calibrate"):
        method.predict_sets(np.random.rand(5, 3))


def test_predict_sets_returns_one_list_per_sample():
    rng = np.random.default_rng(42)
    calibration_probs = rng.dirichlet(np.ones(5), size=200)
    calibration_labels = rng.integers(0, 5, size=200)

    method = APSConformalMethod(alpha=0.1)
    method.calibrate(calibration_probs, calibration_labels)

    test_probs = rng.dirichlet(np.ones(5), size=50)
    sets = method.predict_sets(test_probs)

    assert len(sets) == 50
    assert all(isinstance(s, list) and len(s) >= 1 for s in sets)


def test_empirical_coverage_approximately_matches_target():
    rng = np.random.default_rng(0)
    num_classes = 10
    alpha = 0.1

    def make_probs_and_labels(n):
        labels = rng.integers(0, num_classes, size=n)
        probs = rng.uniform(0.01, 0.2, size=(n, num_classes))
        probs[np.arange(n), labels] += rng.uniform(0.3, 0.8, size=n)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs, labels

    calibration_probs, calibration_labels = make_probs_and_labels(2000)
    test_probs, test_labels = make_probs_and_labels(2000)

    method = APSConformalMethod(alpha=alpha)
    method.calibrate(calibration_probs, calibration_labels)
    sets = method.predict_sets(test_probs)

    covered = [label in s for label, s in zip(test_labels, sets)]
    empirical_coverage = np.mean(covered)

    assert empirical_coverage >= (1 - alpha) - 0.03