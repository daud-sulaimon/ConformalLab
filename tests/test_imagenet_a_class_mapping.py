"""Tests for src.datasets.imagenet_a_class_mapping."""

from src.datasets.imagenet_a_class_mapping import (
    load_imagenet_a_active_indices,
    load_imagenet_a_local_to_full_mapping,
)


def test_active_indices_has_200_entries():
    indices = load_imagenet_a_active_indices()
    assert len(indices) == 200


def test_active_indices_are_within_full_class_range():
    indices = load_imagenet_a_active_indices()
    assert all(0 <= i < 1000 for i in indices)


def test_active_indices_are_unique():
    indices = load_imagenet_a_active_indices()
    assert len(set(indices)) == len(indices)


def test_local_to_full_mapping_has_200_entries():
    mapping = load_imagenet_a_local_to_full_mapping()
    assert len(mapping) == 200


def test_local_to_full_mapping_keys_are_ints():
    mapping = load_imagenet_a_local_to_full_mapping()
    assert all(isinstance(key, int) for key in mapping.keys())