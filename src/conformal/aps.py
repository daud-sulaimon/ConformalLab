"""
Adaptive Prediction Sets (APS) conformal method for ConformalLab.

Romano, Sesia & Candès (2020), "Classification with Valid and Adaptive
Coverage." Unlike LAC, APS adapts prediction-set size to the difficulty
of each input by scoring classes on cumulative sorted probability mass,
rather than a fixed per-class probability threshold.

Supports two modes, selected via the `randomize` constructor flag:
- randomize=False (default): the deterministic/inclusive variant used
  for this project's PRIMARY results throughout Sprint 2. Every
  already-committed experiment used this mode; changing the default
  would silently invalidate the entire archived results set.
- randomize=True: the original formula as specified by Romano et al.,
  including the randomised tie-breaking term. Used only for the
  explicit deterministic-vs-randomised sensitivity check (see
  src/conformal/nonconformity.py's module docstring and
  scripts/run_randomization_sensitivity.py).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.conformal.base import BaseConformalMethod
from src.conformal.nonconformity import (
    aps_scores,
    aps_scores_randomized,
    conformal_quantile,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APSConformalMethod(BaseConformalMethod):
    """
    Adaptive Prediction Sets (Romano, Sesia & Candès, 2020).

    Parameters
    ----------
    alpha
        Miscoverage rate. Must be strictly between 0 and 1.
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
    >>> method = APSConformalMethod(alpha=0.1)
    >>> method.calibrate(calibration_probs, calibration_labels)
    >>> sets = method.predict_sets(test_probs)
    """

    def __init__(self, alpha: float, randomize: bool = False, seed: int = 0) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
        self._alpha = alpha
        self._randomize = randomize
        self._rng = np.random.default_rng(seed)
        self._q_hat: Optional[float] = None

    @property
    def q_hat(self) -> float:
        if self._q_hat is None:
            raise RuntimeError(
                "APSConformalMethod.calibrate() must be called before accessing q_hat."
            )
        return self._q_hat

    def calibrate(self, calibration_probs: np.ndarray, calibration_labels: np.ndarray) -> None:
        if self._randomize:
            scores = aps_scores_randomized(calibration_probs, calibration_labels, self._rng)
        else:
            scores = aps_scores(calibration_probs, calibration_labels)
        self._q_hat = conformal_quantile(scores, self._alpha)
        logger.info(
            f"APS calibrated (randomize={self._randomize}): alpha={self._alpha}, "
            f"n={len(scores)}, q_hat={self._q_hat:.4f}"
        )

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Build prediction sets.

        Deterministic mode: sort classes by descending probability,
        include the prefix of classes whose cumulative probability
        mass does not exceed q_hat.

        Randomised mode: for each test point, draw a fresh u ~
        Uniform(0,1) and include the prefix of classes whose
        (exclusive cumulative mass so far) + u * (this class's own
        probability) does not exceed q_hat - matching the score
        formula used at calibration time.

        Both modes always include at least the top-1 class.
        """
        if self._q_hat is None:
            raise RuntimeError(
                "APSConformalMethod.calibrate() must be called before predict_sets()."
            )

        prediction_sets: List[List[int]] = []

        if self._randomize:
            u_test = self._rng.uniform(0.0, 1.0, size=test_probs.shape[0])
            for i, row in enumerate(test_probs):
                sorted_idx = np.argsort(-row)
                sorted_probs = row[sorted_idx]
                cumsum_exclusive = np.concatenate(([0.0], np.cumsum(sorted_probs)[:-1]))
                scores = cumsum_exclusive + u_test[i] * sorted_probs
                included = scores <= self._q_hat
                if not included.any():
                    included[0] = True
                prediction_sets.append(sorted_idx[included].tolist())
            return prediction_sets

        for row in test_probs:
            sorted_idx = np.argsort(-row)
            cumsum = np.cumsum(row[sorted_idx])
            included = cumsum <= self._q_hat
            if not included.any():
                included[0] = True
            prediction_sets.append(sorted_idx[included].tolist())
        return prediction_sets