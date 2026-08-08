"""
Nonconformity score functions for ConformalLab.

A nonconformity score measures how "unusual" or "wrong" a prediction
is: low scores mean the model was confident and correct, high scores
mean the model was surprised by the true label. Split CP (and its
variants) calibrate a threshold on these scores from held-out
calibration data, then use that threshold to build prediction sets.

This module currently implements the simplest, most common score
(sometimes called "LAC" - least ambiguous set classifier, or the
"1 - softmax" score). Future methods (APS, RAPS) will add their own
score functions here without changing split_cp.py's calibration logic.
"""

from __future__ import annotations

import numpy as np


def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the "1 - probability of true class" nonconformity score.

    Parameters
    ----------
    probs
        Class probabilities, shape ``(num_samples, num_classes)``.
    labels
        True integer class labels, shape ``(num_samples,)``.

    Returns
    -------
    numpy.ndarray
        Nonconformity scores, shape ``(num_samples,)``. Score is
        ``1 - probs[i, labels[i]]`` for each sample ``i``.

    Examples
    --------
    >>> probs = np.array([[0.9, 0.1], [0.3, 0.7]])
    >>> labels = np.array([0, 1])
    >>> lac_scores(probs, labels)
    array([0.1, 0.3])
    """
    true_class_probs = probs[np.arange(len(labels)), labels]
    return 1.0 - true_class_probs