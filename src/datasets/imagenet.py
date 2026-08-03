"""
ImageNet dataset for ConformalLab.

Streams a fixed-size subset of the ImageNet-1k validation split from
Hugging Face (`ILSVRC/imagenet-1k`), deterministically splits it into
calibration and test sets, and exposes them as PyTorch DataLoaders via
the BaseDataset interface.

Only a small, fixed number of images are streamed and materialized in
memory (default 1000) — matching the project's EXP001 scope — rather
than downloading the full ~50,000-image validation split.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.datasets.base import BaseDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

_HF_DATASET_NAME = "ILSVRC/imagenet-1k"
_HF_SPLIT = "validation"

# Generic placeholder preprocessing, used only if no model-specific
# transform is supplied. Real experiments should pass in the exact
# preprocessing function returned by the model wrapper (e.g. CLIP's
# own preprocess), so that images are prepared exactly as that model
# expects. See src/embeddings/extractor.py.
_DEFAULT_TRANSFORM = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class _ImageListDataset(Dataset):
    """
    Thin torch Dataset wrapper around an in-memory list of
    (PIL.Image, label) pairs, applying a transform on access.
    """

    def __init__(
        self,
        items: List[Tuple[Image.Image, int]],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self._items = items
        self._transform = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self._items[index]
        return self._transform(image.convert("RGB")), label


class ImageNetDataset(BaseDataset):
    """
    ImageNet-1k validation subset, streamed from Hugging Face.

    Parameters
    ----------
    subset_size
        Total number of images to stream and use (calibration + test
        combined). Defaults to 1000, matching the project's EXP001 scope.
    calibration_fraction
        Fraction of `subset_size` assigned to the calibration split;
        the remainder becomes the test split. Defaults to 0.5.
    transform
        Preprocessing function applied to each PIL image, producing a
        tensor. If ``None``, a generic ImageNet-style transform is used
        as a placeholder; real experiments should pass the target
        model's own preprocessing function instead.
    batch_size
        Batch size for both DataLoaders. Defaults to 32.
    num_workers
        Number of DataLoader worker processes. Defaults to 0 (main
        process only), which is the safest default on Windows.
    seed
        Seed used to deterministically shuffle and split the streamed
        images into calibration/test. Defaults to 42.

    Examples
    --------
    >>> dataset = ImageNetDataset(subset_size=1000)
    >>> dataset.load()
    >>> calibration = dataset.calibration_loader()
    """

    def __init__(
        self,
        subset_size: int = 1000,
        calibration_fraction: float = 0.5,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        if not (0 < calibration_fraction < 1):
            raise ValueError(
                f"calibration_fraction must be between 0 and 1, "
                f"got {calibration_fraction}."
            )

        self._subset_size = subset_size
        self._calibration_fraction = calibration_fraction
        self._transform = transform or _DEFAULT_TRANSFORM
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed

        self._class_names: List[str] = []
        self._calibration_items: List[Tuple[Image.Image, int]] = []
        self._test_items: List[Tuple[Image.Image, int]] = []
        self._loaded = False

    def load(self) -> None:
        """
        Stream `subset_size` images from the Hugging Face validation
        split and deterministically split them into calibration/test.
        """
        logger.info(
            f"Streaming {self._subset_size} images from "
            f"{_HF_DATASET_NAME} ({_HF_SPLIT} split)..."
        )

        stream = load_dataset(
            _HF_DATASET_NAME, split=_HF_SPLIT, streaming=True
        )
        self._class_names = stream.features["label"].names

        items: List[Tuple[Image.Image, int]] = [
            (example["image"], example["label"])
            for example in stream.take(self._subset_size)
        ]

        if len(items) < self._subset_size:
            logger.warning(
                f"Requested {self._subset_size} images but only "
                f"{len(items)} were available from the stream."
            )

        # Shuffle with a local, seeded Random instance (not the global
        # `random` module) so this split is deterministic and
        # reproducible without side effects on unrelated code that
        # also uses randomness elsewhere in the same run.
        rng = random.Random(self._seed)
        rng.shuffle(items)

        split_index = int(len(items) * self._calibration_fraction)
        self._calibration_items = items[:split_index]
        self._test_items = items[split_index:]
        self._loaded = True

        logger.info(
            f"Loaded {len(self._calibration_items)} calibration images "
            f"and {len(self._test_items)} test images."
        )

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "ImageNetDataset.load() must be called before accessing "
                "calibration_loader(), test_loader(), or class_names()."
            )

    def calibration_loader(self) -> DataLoader:
        self._require_loaded()
        dataset = _ImageListDataset(self._calibration_items, self._transform)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
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

_DATASET_REGISTRY["imagenet"] = ImageNetDataset