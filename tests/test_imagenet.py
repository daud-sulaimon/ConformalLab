"""Tests for src.datasets.imagenet. Requires network access and a
Hugging Face account with ILSVRC/imagenet-1k access granted."""

import pytest
import torch

from src.datasets.imagenet import ImageNetDataset
from src.datasets.manager import DatasetManager


def test_imagenet_dataset_loads_and_splits(loaded_imagenet_dataset):
    calibration_batches = list(loaded_imagenet_dataset.calibration_loader())
    test_batches = list(loaded_imagenet_dataset.test_loader())

    total_calibration = sum(images.shape[0] for images, _ in calibration_batches)
    total_test = sum(images.shape[0] for images, _ in test_batches)

    assert total_calibration == 5
    assert total_test == 5


def test_imagenet_dataset_produces_correct_tensor_shape(loaded_imagenet_dataset):
    images, labels = next(iter(loaded_imagenet_dataset.calibration_loader()))

    assert images.shape[1:] == (3, 224, 224)
    assert isinstance(labels, torch.Tensor)


def test_imagenet_dataset_class_names_available(loaded_imagenet_dataset):
    names = loaded_imagenet_dataset.class_names()
    assert len(names) == 1000
    assert isinstance(names[0], str)


def test_calling_loaders_before_load_raises():
    dataset = ImageNetDataset(subset_size=4)
    with pytest.raises(RuntimeError, match="load"):
        dataset.calibration_loader()


def test_manager_constructs_imagenet_via_registry():
    manager = DatasetManager("imagenet", subset_size=4)
    manager.load()
    assert len(manager.class_names()) == 1000