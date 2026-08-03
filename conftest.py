"""Shared pytest fixtures for ConformalLab's test suite."""

import pytest

from src.datasets.imagenet import ImageNetDataset


@pytest.fixture(scope="session")
def loaded_imagenet_dataset() -> ImageNetDataset:
    """
    A single, shared ImageNetDataset instance streamed once per test
    session, reused across every test that needs real ImageNet data.
    Avoids each test independently re-streaming from Hugging Face.
    """
    dataset = ImageNetDataset(subset_size=10, calibration_fraction=0.5)
    dataset.load()
    return dataset