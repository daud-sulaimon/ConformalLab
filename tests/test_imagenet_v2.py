"""Tests for src.datasets.imagenet_v2. Requires network access."""

import pytest
import torch

from src.datasets.imagenet_v2 import ImageNetV2Dataset

_FAKE_CLASS_NAMES = [f"class_{i}" for i in range(1000)]


def test_imagenet_v2_loads_test_split():
    dataset = ImageNetV2Dataset(class_names=_FAKE_CLASS_NAMES, subset_size=10)
    dataset.load()

    batches = list(dataset.test_loader())
    total = sum(images.shape[0] for images, _ in batches)
    assert total == 10


def test_imagenet_v2_produces_correct_tensor_shape():
    dataset = ImageNetV2Dataset(class_names=_FAKE_CLASS_NAMES, subset_size=4)
    dataset.load()

    images, labels = next(iter(dataset.test_loader()))
    assert images.shape[1:] == (3, 224, 224)
    assert isinstance(labels, torch.Tensor)


def test_imagenet_v2_calibration_loader_raises():
    dataset = ImageNetV2Dataset(class_names=_FAKE_CLASS_NAMES, subset_size=4)
    dataset.load()
    with pytest.raises(NotImplementedError, match="no calibration split"):
        dataset.calibration_loader()


def test_calling_test_loader_before_load_raises():
    dataset = ImageNetV2Dataset(class_names=_FAKE_CLASS_NAMES, subset_size=4)
    with pytest.raises(RuntimeError, match="load"):
        dataset.test_loader()