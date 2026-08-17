"""Tests for src.conformal.factory."""

import pytest

from src.conformal.factory import METHOD_CLASSES, create_conformal_method


def test_create_lac():
    method = create_conformal_method("lac", alpha=0.1)
    assert type(method).__name__ == "SplitConformalMethod"


def test_create_aps():
    method = create_conformal_method("aps", alpha=0.1)
    assert type(method).__name__ == "APSConformalMethod"


def test_create_raps():
    method = create_conformal_method("raps", alpha=0.1)
    assert type(method).__name__ == "RAPSConformalMethod"


def test_create_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown conformal method"):
        create_conformal_method("not_a_real_method", alpha=0.1)


def test_method_classes_has_exactly_three_entries():
    assert set(METHOD_CLASSES.keys()) == {"lac", "aps", "raps"}