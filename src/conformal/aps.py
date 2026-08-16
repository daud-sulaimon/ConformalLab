"""
Adaptive Prediction Sets (APS) conformal method for ConformalLab.

Romano, Sesia & Candès (2020), "Classification with Valid and Adaptive
Coverage." Unlike LAC, APS adapts prediction-set size to the difficulty
of each input by scoring classes on cumulative sorted probability mass,
rather than a fixed per-class probability threshold.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.conformal.base import BaseConformalMethod
from src.conformal.nonconformity import aps_scores, conformal_quantile
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APSConformalMethod(BaseConformalMethod):
    """
    Adaptive Prediction Sets (Romano, Sesia & Candès, 2020).

    Parameters
    ----------
    alpha
        Miscoverage rate. Must be strictly between 0 and 1.

    Examples
    --------
    >>> method = APSConformalMethod(alpha=0.1)
    >>> method.calibrate(calibration_probs, calibration_labels)
    >>> sets = method.predict_sets(test_probs)
    """

    def __init__(self, alpha: float) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
        self._alpha = alpha
        self._q_hat: Optional[float] = None

    @property
    def q_hat(self) -> float:
        if self._q_hat is None:
            raise RuntimeError(
                "APSConformalMethod.calibrate() must be called before accessing q_hat."
            )
        return self._q_hat

    def calibrate(self, calibration_probs: np.ndarray, calibration_labels: np.ndarray) -> None:
        scores = aps_scores(calibration_probs, calibration_labels)
        self._q_hat = conformal_quantile(scores, self._alpha)
        logger.info(
            f"APS calibrated: alpha={self._alpha}, n={len(scores)}, q_hat={self._q_hat:.4f}"
        )

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Build prediction sets: sort classes by descending probability,
        include the prefix of classes whose cumulative probability mass
        does not exceed q_hat. Always includes at least the top-1 class.
        """
        if self._q_hat is None:
            raise RuntimeError(
                "APSConformalMethod.calibrate() must be called before predict_sets()."
            )

        prediction_sets: List[List[int]] = []
        for row in test_probs:
            sorted_idx = np.argsort(-row)
            cumsum = np.cumsum(row[sorted_idx])
            included = cumsum <= self._q_hat
            if not included.any():
                included[0] = True
            prediction_sets.append(sorted_idx[included].tolist())
        return prediction_sets