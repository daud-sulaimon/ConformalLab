"""
Abstract base class for every dataset used in ConformalLab.

Every concrete dataset (ImageNet, ImageNet-R, ImageNet-V2, ImageNet-A)
must inherit from BaseDataset and implement its four methods. This is
the contract that keeps the rest of the framework (models, embeddings,
conformal prediction) completely dataset-agnostic: code written against
BaseDataset works identically regardless of which concrete dataset is
plugged in.

This module contains no data-loading logic of its own — only the
interface definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import DataLoader


class BaseDataset(ABC):
    """
    Abstract interface that every ConformalLab dataset must implement.

    Notes
    -----
    Subclasses are responsible for downloading/validating data,
    applying the correct preprocessing transforms, and producing a
    deterministic calibration/test split (using the project's shared
    seed, see `src.utils.seed`). This class defines *what* every
    dataset must expose, not *how*.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Prepare the dataset for use: validate it exists on disk (or
        download it), and construct the underlying data structures
        needed by `calibration_loader` and `test_loader`.
        """

    @abstractmethod
    def calibration_loader(self) -> DataLoader:
        """
        Return a DataLoader over the calibration split.

        Returns
        -------
        torch.utils.data.DataLoader
            Batches of (image, label) pairs used to calibrate a
            conformal prediction method.
        """

    @abstractmethod
    def test_loader(self) -> DataLoader:
        """
        Return a DataLoader over the test split.

        Returns
        -------
        torch.utils.data.DataLoader
            Batches of (image, label) pairs held out for evaluating
            coverage and set size.
        """

    @abstractmethod
    def class_names(self) -> List[str]:
        """
        Return the ordered list of class names for this dataset.

        Returns
        -------
        list of str
            Class names, ordered to match the integer label indices
            used elsewhere (e.g. index 0 -> class_names()[0]).
        """