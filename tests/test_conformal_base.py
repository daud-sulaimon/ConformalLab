"""Tests for src.conformal.base."""

import pytest

from src.conformal.base import BaseConformalMethod


def test_base_conformal_method_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseConformalMethod()  # abstract methods unimplemented