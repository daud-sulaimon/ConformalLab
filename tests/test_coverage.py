"""Tests for src.metrics.coverage."""

import numpy as np
import pytest

from src.metrics.coverage import average_set_size, coverage_report, empirical_coverage


def test_empirical_coverage_all_covered():
    prediction_sets = [[0, 1], [2, 3], [4]]
    true_labels = np.array([0, 3, 4])
    assert empirical_coverage(prediction_sets, true_labels) == 1.0


def test_empirical_coverage_none_covered():
    prediction_sets = [[1], [2], [3]]
    true_labels = np.array([0, 0, 0])
    assert empirical_coverage(prediction_sets, true_labels) == 0.0


def test_empirical_coverage_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="Mismatched lengths"):
        empirical_coverage([[0], [1]], np.array([0, 1, 2]))


def test_average_set_size():
    prediction_sets = [[0, 1, 2], [3], [4, 5]]
    assert average_set_size(prediction_sets) == pytest.approx(2.0)


def test_coverage_report_contains_expected_keys():
    prediction_sets = [[0], [1], [2]]
    true_labels = np.array([0, 1, 2])
    report = coverage_report(prediction_sets, true_labels, alpha=0.1)

    assert report["target_coverage"] == pytest.approx(0.9)
    assert report["empirical_coverage"] == 1.0
    assert report["average_set_size"] == pytest.approx(1.0)
    assert report["num_samples"] == 3