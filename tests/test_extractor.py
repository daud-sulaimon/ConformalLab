"""Tests for src.embeddings.extractor, using lightweight fakes instead
of real CLIP/ImageNet to keep this test fast and focused on
orchestration logic."""

import numpy as np
import torch

from src.embeddings.cache import EmbeddingCache
from src.embeddings.extractor import extract_embeddings


class _FakeDataset:
    """Minimal stand-in for a BaseDataset: yields two fixed batches."""

    def _make_loader(self):
        images = torch.rand(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        return [(images, labels)]

    def calibration_loader(self):
        return self._make_loader()

    def test_loader(self):
        return self._make_loader()


class _FakeModel:
    """Minimal stand-in for a BaseModel: returns fixed-size embeddings."""

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.rand(batch_size, 8)  # small embedding dim for speed


def test_extract_embeddings_produces_both_splits(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    result = extract_embeddings(
        _FakeDataset(), _FakeModel(), cache_key_prefix="fake", cache=cache
    )

    assert "calibration" in result
    assert "test" in result

    calibration_embeddings, calibration_labels = result["calibration"]
    assert calibration_embeddings.shape == (4, 8)
    assert calibration_labels.shape == (4,)


def test_extract_embeddings_uses_cache_on_second_call(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)

    first_result = extract_embeddings(
        _FakeDataset(), _FakeModel(), cache_key_prefix="fake", cache=cache
    )
    # Second call: even with a *different* fake model instance that
    # would produce different random embeddings, cached values should
    # be returned unchanged, since the cache key already exists.
    second_result = extract_embeddings(
        _FakeDataset(), _FakeModel(), cache_key_prefix="fake", cache=cache
    )

    first_embeddings, _ = first_result["calibration"]
    second_embeddings, _ = second_result["calibration"]
    assert np.array_equal(first_embeddings, second_embeddings)


def test_extract_embeddings_force_recompute_ignores_cache(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)

    first_result = extract_embeddings(
        _FakeDataset(), _FakeModel(), cache_key_prefix="fake", cache=cache
    )
    second_result = extract_embeddings(
        _FakeDataset(),
        _FakeModel(),
        cache_key_prefix="fake",
        cache=cache,
        force_recompute=True,
    )

    first_embeddings, _ = first_result["calibration"]
    second_embeddings, _ = second_result["calibration"]
    # Different random fake embeddings each time model.encode_images is
    # actually called, so forcing recompute should change the values.
    assert not np.array_equal(first_embeddings, second_embeddings)