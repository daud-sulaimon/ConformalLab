"""
Mapping from ImageNet-A's local class indices (0-199, as used by the
clip-benchmark/wds_imagenet-a dataset's `cls` field) to the
corresponding index in ConformalLab's full 1000-class ImageNet
ordering (src/datasets/imagenet_class_names.json).

Per standard practice in the calibration/robustness literature
(Hendrycks et al., 2021, "Natural Adversarial Examples"), evaluation
on ImageNet-A restricts the model's output to only the 200 classes
ImageNet-A covers, rather than the full 1000.

This mapping uses Hendrycks et al.'s own official thousand_k_to_200
index mapping (published in
https://github.com/hendrycks/natural-adv-examples/blob/master/eval.py),
inverted to local->full and verified to align with clip-benchmark's
`cls` ordering. All 200 classes are covered - no exclusions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_MAPPING_PATH = Path(__file__).parent / "imagenet_a_class_mapping.json"


def load_imagenet_a_active_indices() -> List[int]:
    """
    Load the list of full-1000-class indices that ImageNet-A covers.

    Returns
    -------
    list of int
        200 indices into the full 1000-class ordering, sorted
        ascending.

    Raises
    ------
    FileNotFoundError
        If the mapping file doesn't exist.
    """
    if not _MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"{_MAPPING_PATH} not found. This file should be committed "
            f"to the repository; it is not regenerated at runtime."
        )
    with open(_MAPPING_PATH, "r", encoding="utf-8") as f:
        local_to_full: Dict[str, int] = json.load(f)

    return sorted(local_to_full.values())


def load_imagenet_a_local_to_full_mapping() -> Dict[int, int]:
    """
    Load the full local (0-199) -> full-1000-class index mapping.

    Returns
    -------
    dict of int to int
        Keys are ImageNet-A's local `cls` values (0-199), values are
        the corresponding index in the full 1000-class ordering.
    """
    if not _MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"{_MAPPING_PATH} not found. This file should be committed "
            f"to the repository; it is not regenerated at runtime."
        )
    with open(_MAPPING_PATH, "r", encoding="utf-8") as f:
        raw: Dict[str, int] = json.load(f)
    return {int(local_index): full_index for local_index, full_index in raw.items()}