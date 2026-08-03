"""Tests for src.models.base_model."""

import pytest

from src.models.base_model import BaseModel


def test_base_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModel()  # abstract methods unimplemented