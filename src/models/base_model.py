"""
Abstract base class for every Vision-Language Model wrapper in
ConformalLab.

Every concrete model (CLIP, SigLIP) must inherit from BaseModel and
implement its four methods. This is the contract that keeps the rest
of the framework (embeddings, conformal prediction) completely
model-agnostic: code written against BaseModel works identically
regardless of which concrete VLM is plugged in.

This module contains no model-loading or inference logic of its own —
only the interface definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class BaseModel(ABC):
    """
    Abstract interface that every ConformalLab Vision-Language Model
    wrapper must implement.

    Notes
    -----
    Subclasses are responsible for loading model weights, hiding the
    underlying library's API (e.g. OpenCLIP) completely, and exposing
    only this interface. Downstream code should never import
    ``open_clip`` or any model-specific library directly.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and move the model to the appropriate
        device (CPU or GPU).
        """

    @abstractmethod
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of preprocessed images into embeddings.

        Parameters
        ----------
        images
            Batch of preprocessed image tensors, shape
            ``(batch_size, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Image embeddings, shape ``(batch_size, embedding_dim)``.
        """

    @abstractmethod
    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        """
        Encode a list of class names into text embeddings.

        Parameters
        ----------
        class_names
            Ordered list of class name strings (e.g. ImageNet's 1000
            class labels).

        Returns
        -------
        torch.Tensor
            Text embeddings, shape ``(num_classes, embedding_dim)``.
        """

    @abstractmethod
    def predict(self, images: torch.Tensor, class_names: List[str]) -> torch.Tensor:
        """
        Produce class probability scores for a batch of images.

        Parameters
        ----------
        images
            Batch of preprocessed image tensors, shape
            ``(batch_size, channels, height, width)``.
        class_names
            Ordered list of class name strings, defining the label
            space and the order of the returned probability columns.

        Returns
        -------
        torch.Tensor
            Class probabilities, shape ``(batch_size, num_classes)``,
            each row summing to 1.
        """