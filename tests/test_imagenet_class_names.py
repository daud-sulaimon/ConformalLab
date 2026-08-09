"""Tests for src.datasets.imagenet_class_names."""

from src.datasets.imagenet_class_names import load_imagenet_class_names


def test_loads_1000_class_names():
    names = load_imagenet_class_names()
    assert len(names) == 1000
    assert isinstance(names[0], str)