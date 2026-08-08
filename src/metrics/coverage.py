"""
Coverage and set-size metrics for ConformalLab.

Empirical coverage measures whether a conformal method's theoretical
guarantee actually holds on real data: the fraction of test samples
whose prediction set contains the true label. Average set size is the
necessary companion metric — a method that always predicts every
class trivially achieves 100% coverage but provides no useful
information, so coverage must always be reported alongside set size.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def empirical_coverage(prediction_sets: List[List[int]], true_labels: np.ndarray) -> float:
    """
    Compute the fraction of samples whose prediction set contains the
    true label.

    Parameters
    ----------
    prediction_sets
        One list of included class indices per sample, as returned by
        `BaseConformalMethod.predict_sets`.
    true_labels
        True integer class labels, shape ``(num_samples,)``.

    Returns
    -------
    float
        Fraction of samples covered, between 0 and 1.

    Raises
    ------
    ValueError
        If `prediction_sets` and `true_labels` have mismatched lengths.
    """
    if len(prediction_sets) != len(true_labels):
        raise ValueError(
            f"Mismatched lengths: {len(prediction_sets)} prediction sets, "
            f"{len(true_labels)} labels."
        )

    covered = [
        label in prediction_set
        for label, prediction_set in zip(true_labels, prediction_sets)
    ]
    return float(np.mean(covered))


def average_set_size(prediction_sets: List[List[int]]) -> float:
    """
    Compute the average number of classes included per prediction set.

    Parameters
    ----------
    prediction_sets
        One list of included class indices per sample.

    Returns
    -------
    float
        Mean prediction set size across all samples.
    """
    sizes = [len(prediction_set) for prediction_set in prediction_sets]
    return float(np.mean(sizes))


def coverage_report(prediction_sets: List[List[int]], true_labels: np.ndarray, alpha: float) -> dict:
    """
    Build a summary dict combining coverage, set size, and the target.

    Parameters
    ----------
    prediction_sets
        One list of included class indices per sample.
    true_labels
        True integer class labels.
    alpha
        The miscoverage rate the conformal method was calibrated for
        (target coverage is ``1 - alpha``).

    Returns
    -------
    dict
        ``{"target_coverage", "empirical_coverage", "average_set_size", "num_samples"}``.
    """
    coverage = empirical_coverage(prediction_sets, true_labels)
    set_size = average_set_size(prediction_sets)

    report = {
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "average_set_size": set_size,
        "num_samples": len(true_labels),
    }

    logger.info(
        f"Coverage report: target={report['target_coverage']:.3f}, "
        f"empirical={coverage:.3f}, avg_set_size={set_size:.2f}, "
        f"n={report['num_samples']}"
    )

    return report