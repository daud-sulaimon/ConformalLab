"""
Embedding cache for ConformalLab.

Persists extracted embeddings (and their labels) to disk as .npy
files, keyed by a unique string (e.g. "clip_imagenet_calibration"), so
CLIP/SigLIP inference over a dataset only ever needs to run once. All
downstream experiments (Split CP, Adaptive CP, Weighted CP) then read
from this cache instead of re-running the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CACHE_DIR = Path("embeddings")


class EmbeddingCache:
    """
    Save and load (embeddings, labels) pairs to/from disk, keyed by name.

    Parameters
    ----------
    cache_dir
        Directory under which cached files are stored. Defaults to
        ``embeddings/`` at the project root.

    Examples
    --------
    >>> cache = EmbeddingCache()
    >>> cache.save("clip_imagenet_calibration", embeddings, labels)
    >>> embeddings, labels = cache.load("clip_imagenet_calibration")
    """

    def __init__(self, cache_dir: Path = _DEFAULT_CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _embeddings_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}_embeddings.npy"

    def _labels_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}_labels.npy"

    def exists(self, key: str) -> bool:
        """Return True if both the embeddings and labels files for `key` exist."""
        return self._embeddings_path(key).exists() and self._labels_path(key).exists()

    def save(self, key: str, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """
        Save embeddings and labels to disk under `key`.

        Parameters
        ----------
        key
            Unique identifier for this embedding set, e.g.
            ``"clip_imagenet_calibration"``.
        embeddings
            Array of shape ``(num_samples, embedding_dim)``.
        labels
            Array of shape ``(num_samples,)``.

        Raises
        ------
        ValueError
            If `embeddings` and `labels` have mismatched sample counts.
        """
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatched sample counts: embeddings has "
                f"{embeddings.shape[0]} rows, labels has {labels.shape[0]}."
            )

        np.save(self._embeddings_path(key), embeddings)
        np.save(self._labels_path(key), labels)

        logger.info(
            f"Cached embeddings '{key}': shape={embeddings.shape}, "
            f"saved to {self._cache_dir}"
        )

    def load(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings and labels from disk for `key`.

        Returns
        -------
        tuple of numpy.ndarray
            ``(embeddings, labels)``.

        Raises
        ------
        FileNotFoundError
            If no cached files exist for `key`.
        """
        if not self.exists(key):
            raise FileNotFoundError(
                f"No cached embeddings found for key '{key}' in "
                f"{self._cache_dir}. Run extraction first."
            )

        embeddings = np.load(self._embeddings_path(key))
        labels = np.load(self._labels_path(key))

        logger.info(f"Loaded cached embeddings '{key}': shape={embeddings.shape}")
        return embeddings, labels