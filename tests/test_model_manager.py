"""Tests for src.models.manager."""

import pytest

from src.models.manager import ModelManager


def test_manager_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        ModelManager("not_a_real_model")


def test_manager_constructs_clip_via_registry():
    manager = ModelManager("clip")
    assert manager._model.__class__.__name__ == "CLIPModel"