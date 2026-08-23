"""
Nonconformity score functions and shared calibration-threshold logic
for ConformalLab.

Implements three deterministic score functions:
- lac_scores: 1 - softmax probability of the true class (Split CP / LAC).
- aps_scores: cumulative probability mass of all classes at least as
  likely as the true class (Romano, Sesia & Candès, 2020, deterministic
  / "inclusive" variant - see note below).
- raps_scores: APS score plus a rank-based regularisation penalty
  (Angelopoulos, Bates, Malik & Jordan, 2020).

And two randomised variants, matching the original score formulas as
specified in the source papers (with a uniform random tie-breaking
term), used only for the explicit sensitivity check comparing
deterministic vs. randomised APS/RAPS:
- aps_scores_randomized
- raps_scores_randomized

conformal_quantile() implements the ceil((n+1)(1-alpha))/n quantile
formula shared by every calibration procedure in this project.

IMPORTANT NOTE on aps_scores / raps_scores (the deterministic
variants): these are this project's PRIMARY, already-run experiment
results throughout Sprint 2 (Sections 7.1-7.4, 8.1-8.6 of the
dissertation). They use a simpler formula than Romano et al.'s
original specification: aps_scores sums ALL probabilities >= the true
class's probability (an "inclusive" cumulative sum, no random tie-
break term at all), rather than the original formula's strict
sum-of-greater-probabilities plus a randomised fraction of the true
class's own probability. This is a documented simplification common in
practical implementations (e.g. MAPIE's default APS mode), but it is
NOT the same score as the literature's randomised APS/RAPS, and this
difference plausibly explains part of this project's very large
observed APS set sizes. aps_scores_randomized / raps_scores_randomized
implement the original formula for a direct sensitivity comparison
(see APSConformalMethod's `randomize` flag and
scripts/run_randomization_sensitivity.py). The deterministic variant
remains the PRIMARY result set unless and until the sensitivity check
demonstrates the randomised variant should replace it - that decision
is made explicitly, not silently.
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
    Compute the deterministic/inclusive APS nonconformity score:
    cumulative probability mass of every class at least as likely as
    the true class. See module docstring for how this differs from
    the original randomised formula.

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
    Compute the deterministic RAPS nonconformity score: the
    deterministic APS cumulative score plus a regularisation penalty
    on classes ranked beyond `k_reg`.

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


def aps_scores_randomized(
    probs: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    APS nonconformity score with randomised tie-breaking, per Romano,
    Sesia & Candès (2020): sum of probability mass STRICTLY greater
    than the true class's probability, plus a uniform random fraction
    of the true class's own probability.

    E(x, y; u) = sum_{j: pi_j > pi_y} pi_j + u * pi_y,  u ~ Uniform(0,1)

    This is the score formula as originally specified in the source
    paper, in contrast to `aps_scores` (this project's primary,
    deterministic/inclusive variant - see module docstring). Used only
    for the explicit deterministic-vs-randomised sensitivity check.

    Parameters
    ----------
    probs
        Class probabilities, shape ``(num_samples, num_classes)``.
    labels
        True integer class labels, shape ``(num_samples,)``.
    rng
        A numpy random Generator, passed explicitly (not module-level
        global state) so results are reproducible given a fixed seed.

    Returns
    -------
    numpy.ndarray
        Nonconformity scores, shape ``(num_samples,)``.
    """
    n = probs.shape[0]
    scores = np.empty(n)
    u = rng.uniform(0.0, 1.0, size=n)
    for i in range(n):
        row = probs[i]
        true_prob = row[labels[i]]
        strictly_greater_mass = row[row > true_prob].sum()
        scores[i] = strictly_greater_mass + u[i] * true_prob
    return scores


def raps_scores_randomized(
    probs: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    lam: float = 0.01,
    k_reg: int = 5,
) -> np.ndarray:
    """
    RAPS nonconformity score with randomised tie-breaking, per
    Angelopoulos, Bates, Malik & Jordan (2020): the randomised APS
    score (see `aps_scores_randomized`) plus the same rank-based
    regularisation penalty as the deterministic RAPS variant. Used
    only for the explicit deterministic-vs-randomised sensitivity
    check.

    Parameters
    ----------
    probs
        Class probabilities, shape ``(num_samples, num_classes)``.
    labels
        True integer class labels, shape ``(num_samples,)``.
    rng
        A numpy random Generator, passed explicitly for reproducibility.
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
    u = rng.uniform(0.0, 1.0, size=n)
    for i in range(n):
        row = probs[i]
        sorted_idx = np.argsort(-row)
        true_label = labels[i]
        rank = int(np.where(sorted_idx == true_label)[0][0]) + 1  # 1-indexed
        true_prob = row[true_label]
        strictly_greater_mass = row[row > true_prob].sum()
        base_score = strictly_greater_mass + u[i] * true_prob
        penalty = lam * max(0, rank - k_reg)
        scores[i] = base_score + penalty
    return scores


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the calibration threshold q_hat: the
    ceil((n+1)(1-alpha))/n quantile of nonconformity scores.

    This specific formula (accounting for the test point being
    exchangeable with calibration points), rather than a plain
    n-based quantile, is what makes the marginal coverage guarantee
    hold exactly rather than approximately. Shared by every conformal
    method in this project for consistency.

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