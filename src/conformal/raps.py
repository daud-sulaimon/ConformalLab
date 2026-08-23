"""
Regularized Adaptive Prediction Sets (RAPS) conformal method for
ConformalLab.

Angelopoulos, Bates, Malik & Jordan (2020), "Uncertainty Sets for Image
Classifiers using Conformal Prediction." Extends APS with a rank-based
regularisation penalty that discourages including many low-probability
tail classes, typically producing smaller, more stable prediction sets
than APS at a small cost to adaptivity.

Supports two modes, selected via the `randomize` constructor flag:
- randomize=False (default): the deterministic/inclusive variant used
  for this project's PRIMARY results throughout Sprint 2. Every
  already-committed experiment used this mode; changing the default
  would silently invalidate the entire archived results set.
- randomize=True: the original formula as specified by Angelopoulos et
  al., including the randomised tie-breaking term inherited from APS.
  Used only for the explicit deterministic-vs-randomised sensitivity
  check (see src/conformal/nonconformity.py's module docstring and
  scripts/run_randomization_sensitivity.py).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.conformal.base import BaseConformalMethod
from src.conformal.nonconformity import (
    conformal_quantile,
    raps_scores,
    raps_scores_randomized,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_LAMBDA = 0.01
_DEFAULT_K_REG = 5


class RAPSConformalMethod(BaseConformalMethod):
    """
    Regularized Adaptive Prediction Sets (Angelopoulos et al., 2020).

    Parameters
    ----------
    alpha
        Miscoverage rate. Must be strictly between 0 and 1.
    lam
        Regularisation weight. Defaults to 0.01.
    k_reg
        Rank beyond which the regularisation penalty applies. Defaults
        to 5.
    randomize
        If False (default), use the deterministic/inclusive score
        variant (this project's primary results). If True, use the
        original randomised-tie-break formula, for sensitivity
        comparison only.
    seed
        Seed for the random number generator used when
        ``randomize=True``. Ignored when ``randomize=False``.

    Examples
    --------
    >>> method = RAPSConformalMethod(alpha=0.1)
    >>> method.calibrate(calibration_probs, calibration_labels)
    >>> sets = method.predict_sets(test_probs)
    """

    def __init__(
        self,
        alpha: float,
        lam: float = _DEFAULT_LAMBDA,
        k_reg: int = _DEFAULT_K_REG,
        randomize: bool = False,
        seed: int = 0,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
        self._alpha = alpha
        self._lam = lam
        self._k_reg = k_reg
        self._randomize = randomize
        self._rng = np.random.default_rng(seed)
        self._q_hat: Optional[float] = None

    @property
    def q_hat(self) -> float:
        if self._q_hat is None:
            raise RuntimeError(
                "RAPSConformalMethod.calibrate() must be called before accessing q_hat."
            )
        return self._q_hat

    def calibrate(self, calibration_probs: np.ndarray, calibration_labels: np.ndarray) -> None:
        if self._randomize:
            scores = raps_scores_randomized(
                calibration_probs,
                calibration_labels,
                self._rng,
                lam=self._lam,
                k_reg=self._k_reg,
            )
        else:
            scores = raps_scores(
                calibration_probs, calibration_labels, lam=self._lam, k_reg=self._k_reg
            )
        self._q_hat = conformal_quantile(scores, self._alpha)
        logger.info(
            f"RAPS calibrated (randomize={self._randomize}): alpha={self._alpha}, "
            f"lam={self._lam}, k_reg={self._k_reg}, n={len(scores)}, q_hat={self._q_hat:.4f}"
        )

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Build prediction sets.

        Deterministic mode: sort classes by descending probability,
        include the prefix whose (cumulative mass + rank penalty) does
        not exceed q_hat.

        Randomised mode: for each test point, draw a fresh u ~
        Uniform(0,1) and include the prefix whose (exclusive
        cumulative mass so far + u * this class's own probability +
        rank penalty) does not exceed q_hat - matching the score
        formula used at calibration time.

        Both modes always include at least the top-1 class.
        """
        if self._q_hat is None:
            raise RuntimeError(
                "RAPSConformalMethod.calibrate() must be called before predict_sets()."
            )

        prediction_sets: List[List[int]] = []

        if self._randomize:
            u_test = self._rng.uniform(0.0, 1.0, size=test_probs.shape[0])
            for i, row in enumerate(test_probs):
                sorted_idx = np.argsort(-row)
                sorted_probs = row[sorted_idx]
                cumsum_exclusive = np.concatenate(([0.0], np.cumsum(sorted_probs)[:-1]))
                ranks = np.arange(1, len(row) + 1)
                penalty = self._lam * np.maximum(0, ranks - self._k_reg)
                scores = cumsum_exclusive + u_test[i] * sorted_probs + penalty

                included = scores <= self._q_hat
                if not included.any():
                    included[0] = True
                prediction_sets.append(sorted_idx[included].tolist())
            return prediction_sets

        for row in test_probs:
            sorted_idx = np.argsort(-row)
            sorted_probs = row[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            ranks = np.arange(1, len(row) + 1)
            penalty = self._lam * np.maximum(0, ranks - self._k_reg)
            scores = cumsum + penalty

            included = scores <= self._q_hat
            if not included.any():
                included[0] = True
            prediction_sets.append(sorted_idx[included].tolist())
        return prediction_sets