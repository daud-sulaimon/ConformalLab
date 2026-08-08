"""
Split Conformal Prediction for ConformalLab.

Implements the standard two-step Split CP procedure:

1. calibrate(): compute a single threshold q_hat from calibration-set
   nonconformity scores, using the (n+1)(1-alpha)/n quantile formula
   required for the marginal coverage guarantee to hold exactly.
2. predict_sets(): for each test sample, include every class whose
   nonconformity score does not exceed q_hat.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.conformal.base import BaseConformalMethod
from src.conformal.nonconformity import lac_scores
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SplitConformalMethod(BaseConformalMethod):
    """
    Split Conformal Prediction using the LAC (1 - softmax) nonconformity
    score.

    Parameters
    ----------
    alpha
        Miscoverage rate. Prediction sets are guaranteed (under
        exchangeability) to contain the true label with probability
        at least ``1 - alpha``. Must be strictly between 0 and 1.

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

    def calibrate(self, calibration_probs: np.ndarray, calibration_labels: np.ndarray) -> None:
        """
        Compute the calibration threshold q_hat from calibration data.

        Uses the ceil((n+1)(1-alpha)) / n quantile of calibration
        nonconformity scores, rather than a plain n-based quantile —
        this specific formula (accounting for the test point being
        exchangeable with calibration points) is what makes the
        marginal coverage guarantee hold exactly rather than
        approximately.
        """
        scores = lac_scores(calibration_probs, calibration_labels)
        n = len(scores)

        quantile_rank = math.ceil((n + 1) * (1 - self._alpha))
        quantile_level = min(quantile_rank / n, 1.0)

        self._q_hat = float(np.quantile(scores, quantile_level, method="higher"))

        logger.info(
            f"Split CP calibrated: alpha={self._alpha}, n={n}, "
            f"quantile_level={quantile_level:.4f}, q_hat={self._q_hat:.4f}"
        )

    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Build a prediction set for each test sample: every class whose
        (1 - probability) does not exceed q_hat.

        Raises
        ------
        RuntimeError
            If `calibrate` has not been called yet.
        """
        if self._q_hat is None:
            raise RuntimeError(
                "SplitConformalMethod.calibrate() must be called before predict_sets()."
            )

        # A class is included if its nonconformity score (1 - prob) is
        # <= q_hat, i.e. if prob >= 1 - q_hat.
        threshold_prob = 1.0 - self._q_hat
        included = test_probs >= threshold_prob

        return [np.where(row)[0].tolist() for row in included]