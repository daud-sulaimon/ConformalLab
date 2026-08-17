"""
Factory for constructing conformal methods by name.

Centralises the name -> class mapping shared by every experiment
script (run_split_cp.py, run_shift_eval.py,
run_recalibration_sweep.py), so adding a new conformal method only
requires updating this one file, not each script independently.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from src.conformal.aps import APSConformalMethod
from src.conformal.base import BaseConformalMethod
from src.conformal.raps import RAPSConformalMethod
from src.conformal.split_cp import SplitConformalMethod

METHOD_CLASSES: Dict[str, Type[BaseConformalMethod]] = {
    "lac": SplitConformalMethod,
    "aps": APSConformalMethod,
    "raps": RAPSConformalMethod,
}


def create_conformal_method(name: str, **kwargs: Any) -> BaseConformalMethod:
    """
    Construct a conformal method by name.

    Parameters
    ----------
    name
        Method identifier: "lac", "aps", or "raps".
    **kwargs
        Forwarded to the underlying method's constructor (e.g. `alpha`,
        and for RAPS optionally `lam`/`k_reg`).

    Returns
    -------
    BaseConformalMethod
        A constructed, uncalibrated conformal method instance.

    Raises
    ------
    ValueError
        If `name` is not a recognised method.
    """
    if name not in METHOD_CLASSES:
        available = sorted(METHOD_CLASSES.keys())
        raise ValueError(f"Unknown conformal method '{name}'. Available: {available}")
    return METHOD_CLASSES[name](**kwargs)