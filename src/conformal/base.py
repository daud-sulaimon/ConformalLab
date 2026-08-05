"""
Abstract base class for every conformal prediction method in
ConformalLab.

Every concrete method (Split CP, Adaptive CP, Weighted CP) must
inherit from BaseConformalMethod and implement its two methods. This
is the contract that keeps evaluation code (coverage, set size
metrics) completely agnostic to which specific CP method produced the
prediction sets.

This module contains no calibration logic of its own — only the
interface definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseConformalMethod(ABC):
    """
    Abstract interface that every ConformalLab conformal prediction
    method must implement.

    Notes
    -----
    Subclasses are responsible for computing a calibration threshold
    from calibration-set probabilities/labels, and using that
    threshold to construct prediction sets for test-set probabilities.
    This class defines *what* every method must expose, not *how*.
    """

    @abstractmethod
    def calibrate(self, calibration_probs: np.ndarray, calibration_labels: np.ndarray) -> None:
        """
        Compute and store whatever threshold(s) this method needs,
        using calibration data only.

        Parameters
        ----------
        calibration_probs
            Class probabilities for the calibration set, shape
            ``(num_calibration_samples, num_classes)``.
        calibration_labels
            True integer class labels for the calibration set, shape
            ``(num_calibration_samples,)``.
        """

    @abstractmethod
    def predict_sets(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Construct a prediction set for each test sample.

        Parameters
        ----------
        test_probs
            Class probabilities for the test set, shape
            ``(num_test_samples, num_classes)``.

        Returns
        -------
        list of list of int
            One list per test sample, containing the integer class
            indices included in that sample's prediction set.
        """