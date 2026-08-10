"""Tests for src.datasets.imagenet_a. Requires network access."""

import json

import pytest
import torch

from src.datasets.imagenet_a import ImageNetADataset

with open("src/datasets/imagenet_class_names.json", "r", encoding="utf-8") as f:
    _FULL_CLASS_NAMES = json.load(f)


def test_imagenet_a_loads_test_split():
    dataset = ImageNetADataset(class_names=_FULL_CLASS_NAMES, subset_size=10)
    dataset.load()

    batches = list(dataset.test_loader())
    total = sum(images.shape[0] for images, _ in batches)
    assert total == 10


def test_imagenet_a_class_names_has_200_entries():
    dataset = ImageNetADataset(class_names=_FULL_CLASS_NAMES, subset_size=4)
    dataset.load()
    assert len(dataset.class_names()) == 200


def test_imagenet_a_produces_correct_tensor_shape():
    dataset = ImageNetADataset(class_names=_FULL_CLASS_NAMES, subset_size=4)
    dataset.load()

    images, labels = next(iter(dataset.test_loader()))
    assert images.shape[1:] == (3, 224, 224)
    assert isinstance(labels, torch.Tensor)
    assert torch.all(labels < 200)


def test_imagenet_a_calibration_loader_raises():
    dataset = ImageNetADataset(class_names=_FULL_CLASS_NAMES, subset_size=4)
    dataset.load()
    with pytest.raises(NotImplementedError, match="no calibration split"):
        dataset.calibration_loader()


def test_calling_test_loader_before_load_raises():
    dataset = ImageNetADataset(class_names=_FULL_CLASS_NAMES, subset_size=4)
    with pytest.raises(RuntimeError, match="load"):
        dataset.test_loader()