"""
ImageNet-V2 dataset for ConformalLab.

Streams a fixed-size subset of ImageNet-V2 (Recht et al., 2019) from
the clip-benchmark WebDataset mirror on Hugging Face. Unlike
ImageNetDataset, this dataset provides only a test split — it is used
to evaluate a conformal method already calibrated on ImageNet, not to
recalibrate. ImageNet-V2 was constructed to preserve ImageNet's
original 1000-class label space and ordering, so `cls` indices here
correspond directly to the same class_names ordering as ImageNet.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.base import BaseDataset
from src.datasets.imagenet import _DEFAULT_TRANSFORM, _ImageListDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

_HF_DATASET_NAME = "clip-benchmark/wds_imagenetv2"
_HF_SPLIT = "test"


class ImageNetV2Dataset(BaseDataset):
    """
    ImageNet-V2 (Recht et al., 2019), streamed from Hugging Face.

    A mild natural distribution shift benchmark: a new sample of the
    same 1000 ImageNet classes, collected roughly a decade after the
    original ImageNet validation set. Provides only a test split;
    calibration_loader() is intentionally unsupported, since ordinary
    Split CP evaluation calibrates on ImageNet and only evaluates
    coverage on the shifted set.

    Parameters
    ----------
    subset_size
        Number of images to stream for the test split. Defaults to 500,
        matching ImageNetDataset's test split size for direct comparison.
    class_names
        The class name list from the calibration dataset (ImageNet),
        required since this dataset does not expose its own class
        name mapping - only integer `cls` values matching that ordering.
    transform
        Preprocessing function, typically the target model's own
        preprocessing (e.g. CLIP's). Defaults to a generic placeholder.
    batch_size, num_workers, seed
        As in ImageNetDataset.

    Examples
    --------
    >>> dataset = ImageNetV2Dataset(subset_size=500, class_names=imagenet_class_names)
    >>> dataset.load()
    >>> test = dataset.test_loader()
    """

    def __init__(
        self,
        class_names: List[str],
        subset_size: int = 500,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        self._class_names = class_names
        self._subset_size = subset_size
        self._transform = transform or _DEFAULT_TRANSFORM
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed

        self._test_items: List[Tuple[Image.Image, int]] = []
        self._loaded = False

    def load(self) -> None:
        logger.info(
            f"Streaming {self._subset_size} images from "
            f"{_HF_DATASET_NAME} ({_HF_SPLIT} split)..."
        )

        stream = load_dataset(_HF_DATASET_NAME, split=_HF_SPLIT, streaming=True)

        items: List[Tuple[Image.Image, int]] = [
            (example["webp"], example["cls"])
            for example in stream.take(self._subset_size)
        ]

        if len(items) < self._subset_size:
            logger.warning(
                f"Requested {self._subset_size} images but only "
                f"{len(items)} were available from the stream."
            )

        rng = random.Random(self._seed)
        rng.shuffle(items)

        self._test_items = items
        self._loaded = True

        logger.info(f"Loaded {len(self._test_items)} test images from ImageNet-V2.")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "ImageNetV2Dataset.load() must be called before accessing "
                "test_loader() or class_names()."
            )

    def calibration_loader(self) -> DataLoader:
        raise NotImplementedError(
            "ImageNetV2Dataset has no calibration split. Calibrate a "
            "conformal method on ImageNetDataset, then evaluate coverage "
            "on this dataset's test_loader()."
        )

    def test_loader(self) -> DataLoader:
        self._require_loaded()
        dataset = _ImageListDataset(self._test_items, self._transform)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def class_names(self) -> List[str]:
        self._require_loaded()
        return self._class_names


from src.datasets.manager import _DATASET_REGISTRY

_DATASET_REGISTRY["imagenet_v2"] = ImageNetV2Dataset