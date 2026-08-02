"""Tests for src.datasets.base and src.datasets.manager."""

import pytest

from src.datasets.base import BaseDataset
from src.datasets.manager import DatasetManager


def test_base_dataset_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseDataset()  # abstract methods unimplemented


def test_manager_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        DatasetManager("not_a_real_dataset")