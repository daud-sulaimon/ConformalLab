"""
Nonconformity score functions and shared calibration-threshold logic
for ConformalLab.

Implements three score functions:
- lac_scores: 1 - softmax probability of the true class (Split CP / LAC).
- aps_scores: cumulative probability mass of all classes at least as
  likely as the true class (Romano, Sesia & Candès, 2020).
- raps_scores: APS score plus a rank-based regularisation penalty
  (Angelopoulos, Bates, Malik & Jordan, 2020).

conformal_quantile() implements the ceil((n+1)(1-alpha))/n quantile
formula shared by all three calibration procedures, factored into one
place so every conformal method uses bit-identical threshold logic -
important since RQ2 directly compares these three methods.
"""

from __future__ import annotations

import math

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
        Nonconformity scores, shape ``(num_samples,)``.
    """
    true_class_probs = probs[np.arange(len(labels)), labels]
    return 1.0 - true_class_probs


def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the APS nonconformity score: cumulative probability mass
    of every class at least as likely as the true class.

    Parameters
    ----------
    probs
        Class probabilities, shape ``(num_samples, num_classes)``.
    labels
        True integer class labels, shape ``(num_samples,)``.

    Returns
    -------
    numpy.ndarray
        Nonconformity scores, shape ``(num_samples,)``.
    """
    n = probs.shape[0]
    scores = np.empty(n)
    for i in range(n):
        row = probs[i]
        true_prob = row[labels[i]]
        scores[i] = row[row >= true_prob].sum()
    return scores


def raps_scores(
    probs: np.ndarray, labels: np.ndarray, lam: float = 0.01, k_reg: int = 5
) -> np.ndarray:
    """
    Compute the RAPS nonconformity score: the APS cumulative score plus
    a regularisation penalty on classes ranked beyond `k_reg`.

    Parameters
    ----------
    probs
        Class probabilities, shape ``(num_samples, num_classes)``.
    labels
        True integer class labels, shape ``(num_samples,)``.
    lam
        Regularisation weight. Defaults to 0.01.
    k_reg
        Rank beyond which the penalty applies. Defaults to 5.

    Returns
    -------
    numpy.ndarray
        Nonconformity scores, shape ``(num_samples,)``.
    """
    n = probs.shape[0]
    scores = np.empty(n)
    for i in range(n):
        row = probs[i]
        sorted_idx = np.argsort(-row)
        sorted_probs = row[sorted_idx]
        cumsum = np.cumsum(sorted_probs)

        true_label = labels[i]
        rank = int(np.where(sorted_idx == true_label)[0][0]) + 1  # 1-indexed
        base_score = cumsum[rank - 1]
        penalty = lam * max(0, rank - k_reg)
        scores[i] = base_score + penalty
    return scores


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the calibration threshold q_hat: the
    ceil((n+1)(1-alpha))/n quantile of nonconformity scores.

    Parameters
    ----------
    scores
        Calibration nonconformity scores.
    alpha
        Miscoverage rate.

    Returns
    -------
    float
        The calibration threshold q_hat.
    """
    n = len(scores)
    quantile_rank = math.ceil((n + 1) * (1 - alpha))
    quantile_level = min(quantile_rank / n, 1.0)
    return float(np.quantile(scores, quantile_level, method="higher"))