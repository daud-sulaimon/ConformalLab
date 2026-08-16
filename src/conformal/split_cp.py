"""
Split Conformal Prediction (LAC) for ConformalLab.

Implements the standard two-step Split CP procedure using the shared
conformal_quantile() calibration logic (see nonconformity.py) so its
threshold computation is guaranteed identical to APS/RAPS.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.conformal.base import BaseConformalMethod
from src.conformal.nonconformity import conformal_quantile, lac_scores
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SplitConformalMethod(BaseConformalMethod):
    """
    Split Conformal Prediction using the LAC (1 - softmax) nonconformity
    score.

    Parameters
    ----------
    alpha
        Miscoverage rate. Must be strictly between 0 and 1.

    Examples
    --------
    >>> method = SplitConformalMethod(alpha=0.1)
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
        """The calibrated threshold. Raises if calibrate() hasn't been called."""
        if self._q_hat is None:
            raise RuntimeError(
                "SplitConformalMethod.calibrate() must be called before accessing q_hat."
            )
        return self._q_hat

    def calibrate(self, calibration_probs: np.ndarray, calibration_labels: np.ndarray) -> None:
        scores = lac_scores(calibration_probs, calibration_labels)
        self._q_hat = conformal_quantile(scores, self._alpha)
        logger.info(
            f"Split CP calibrated: alpha={self._alpha}, n={len(scores)}, q_hat={self._q_hat:.4f}"
        )

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        if self._q_hat is None:
            raise RuntimeError(
                "SplitConformalMethod.calibrate() must be called before predict_sets()."
            )
        threshold_prob = 1.0 - self._q_hat
        included = test_probs >= threshold_prob
        return [np.where(row)[0].tolist() for row in included]