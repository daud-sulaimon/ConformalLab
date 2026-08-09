"""
Static loader for ImageNet's 1000 class names.

Reads from a committed JSON file (generated once via
scripts/generate_imagenet_classnames.py) rather than streaming from
Hugging Face - class names are fixed, deterministic dataset metadata
shared by ImageNet and every shift dataset (V2, R, A) that uses the
same 1000-class label space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

_CLASS_NAMES_PATH = Path(__file__).parent / "imagenet_class_names.json"


def load_imagenet_class_names() -> List[str]:
    """
    Load ImageNet's 1000 class names from the committed JSON file.

    Returns
    -------
    list of str
        1000 class names, in the same order as ImageNet's integer
        label indices.

    Raises
    ------
    FileNotFoundError
        If the class names file doesn't exist - run
        scripts/generate_imagenet_classnames.py first.
    """
    if not _CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"{_CLASS_NAMES_PATH} not found. Run "
            f"scripts/generate_imagenet_classnames.py first."
        )
    with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)