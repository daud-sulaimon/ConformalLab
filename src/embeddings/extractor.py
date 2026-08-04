"""
Embedding extraction pipeline for ConformalLab.

Ties together a dataset, a model, and the embedding cache: runs every
image in a DataLoader through the model to produce embeddings, checks
the cache first to avoid redundant computation, and saves newly
computed embeddings for future reuse.

This is the only module that should coordinate a BaseDataset and a
BaseModel together — neither of those layers know about each other
directly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.base import BaseDataset
from src.embeddings.cache import EmbeddingCache
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_from_loader(
    loader: DataLoader, model: BaseModel
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run every batch in `loader` through `model.encode_images`, collecting
    embeddings and labels.

    Returns
    -------
    tuple of numpy.ndarray
        ``(embeddings, labels)``, both with one row per image, in the
        same order the loader produced them.
    """
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        embeddings = model.encode_images(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


def extract_embeddings(
    dataset: BaseDataset,
    model: BaseModel,
    cache_key_prefix: str,
    cache: EmbeddingCache | None = None,
    force_recompute: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Extract (or load cached) embeddings for a dataset's calibration and
    test splits.

    Parameters
    ----------
    dataset
        A loaded `BaseDataset` instance (`.load()` already called).
    model
        A loaded `BaseModel` instance (`.load()` already called).
    cache_key_prefix
        Prefix used to build cache keys, e.g. ``"clip_imagenet"``
        produces cache keys ``"clip_imagenet_calibration"`` and
        ``"clip_imagenet_test"``.
    cache
        `EmbeddingCache` instance to use. If ``None``, a default
        instance (caching to ``embeddings/`` at the project root) is
        created.
    force_recompute
        If ``True``, ignore any existing cache entry and recompute.
        Defaults to ``False``.

    Returns
    -------
    dict
        ``{"calibration": (embeddings, labels), "test": (embeddings, labels)}``.

    Examples
    --------
    >>> dataset = DatasetManager("imagenet", subset_size=1000)
    >>> dataset.load()
    >>> model = ModelManager("clip")
    >>> model.load()
    >>> result = extract_embeddings(dataset, model, cache_key_prefix="clip_imagenet")
    >>> calibration_embeddings, calibration_labels = result["calibration"]
    """
    if cache is None:
        cache = EmbeddingCache()

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split_name, loader_fn in [
        ("calibration", dataset.calibration_loader),
        ("test", dataset.test_loader),
    ]:
        cache_key = f"{cache_key_prefix}_{split_name}"

        if not force_recompute and cache.exists(cache_key):
            logger.info(f"Using cached embeddings for '{cache_key}'.")
            results[split_name] = cache.load(cache_key)
            continue

        logger.info(f"Extracting embeddings for '{cache_key}'...")
        loader = loader_fn()
        embeddings, labels = _extract_from_loader(loader, model)
        cache.save(cache_key, embeddings, labels)
        results[split_name] = (embeddings, labels)

    return results