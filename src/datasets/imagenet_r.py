"""
ImageNet-R dataset for ConformalLab.

Streams a fixed-size subset of ImageNet-R (Hendrycks et al., 2021)
from the clip-benchmark WebDataset mirror on Hugging Face. ImageNet-R
covers only 200 of ImageNet's 1000 classes (renditions - art,
cartoons, sculptures - of those 200 classes). Following standard
practice in the calibration/robustness literature, this dataset
exposes only those 200 classes via class_names(), with labels
re-indexed into that 200-length space (not the full 1000), using the
authoritative Hendrycks et al. index mapping (see
imagenet_r_class_mapping.py).

Like ImageNetV2Dataset, this dataset provides only a test split - it
evaluates a conformal method already calibrated on full ImageNet, it
does not recalibrate.
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
from src.datasets.imagenet_r_class_mapping import load_imagenet_r_local_to_full_mapping
from src.utils.logger import get_logger

logger = get_logger(__name__)

_HF_DATASET_NAME = "clip-benchmark/wds_imagenet-r"
_HF_SPLIT = "test"


class ImageNetRDataset(BaseDataset):
    """
    ImageNet-R (Hendrycks et al., 2021), streamed from Hugging Face.

    A moderate-to-strong distribution shift benchmark: artistic
    renditions (paintings, cartoons, sculptures, embroidery, etc.) of
    200 of ImageNet's 1000 classes. Provides only a test split.

    Parameters
    ----------
    class_names
        The full 1000-class name list from the calibration dataset
        (ImageNet). This dataset internally restricts to its 200
        relevant classes, but needs the full list to look up their
        names.
    subset_size
        Number of images to stream for the test split. Defaults to 500.
    transform
        Preprocessing function, typically the target model's own
        preprocessing (e.g. CLIP's). Defaults to a generic placeholder.
    batch_size, num_workers, seed
        As in ImageNetDataset.

    Examples
    --------
    >>> dataset = ImageNetRDataset(class_names=imagenet_class_names, subset_size=500)
    >>> dataset.load()
    >>> test = dataset.test_loader()
    >>> restricted_class_names = dataset.class_names()  # 200 names, not 1000
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
        self._full_class_names = class_names
        self._subset_size = subset_size
        self._transform = transform or _DEFAULT_TRANSFORM
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed

        # local_index (0-199) -> full 1000-class index
        self._local_to_full = load_imagenet_r_local_to_full_mapping()
        # The 200 class names, in local-index order, so class_names()
        # returns names whose position matches each item's re-indexed
        # label (see load()).
        self._restricted_class_names = [
            self._full_class_names[self._local_to_full[i]] for i in range(len(self._local_to_full))
        ]

        self._test_items: List[Tuple[Image.Image, int]] = []
        self._loaded = False

    def load(self) -> None:
        logger.info(
            f"Streaming {self._subset_size} images from "
            f"{_HF_DATASET_NAME} ({_HF_SPLIT} split)..."
        )

        stream = load_dataset(_HF_DATASET_NAME, split=_HF_SPLIT, streaming=True)

        # `cls` from the stream is already the local 0-199 index this
        # dataset's class_names() uses - no re-mapping needed at label
        # level, since class_names() itself was built in that same
        # local-index order.
        items: List[Tuple[Image.Image, int]] = [
            (example["jpg"], example["cls"])
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

        logger.info(f"Loaded {len(self._test_items)} test images from ImageNet-R.")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "ImageNetRDataset.load() must be called before accessing "
                "test_loader() or class_names()."
            )

    def calibration_loader(self) -> DataLoader:
        raise NotImplementedError(
            "ImageNetRDataset has no calibration split. Calibrate a "
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
        """
        Returns the 200 class names ImageNet-R covers (not the full
        1000), in the same order as this dataset's re-indexed labels.
        """
        self._require_loaded()
        return self._restricted_class_names


from src.datasets.manager import _DATASET_REGISTRY

_DATASET_REGISTRY["imagenet_r"] = ImageNetRDataset