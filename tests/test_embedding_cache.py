"""Tests for src.embeddings.cache."""

import numpy as np
import pytest

from src.embeddings.cache import EmbeddingCache


def test_save_and_load_roundtrip(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    embeddings = np.random.rand(10, 512).astype(np.float32)
    labels = np.arange(10)

    cache.save("test_key", embeddings, labels)
    loaded_embeddings, loaded_labels = cache.load("test_key")

    assert np.array_equal(embeddings, loaded_embeddings)
    assert np.array_equal(labels, loaded_labels)


def test_exists_returns_false_before_save(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    assert cache.exists("nonexistent_key") is False


def test_exists_returns_true_after_save(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    cache.save("test_key", np.random.rand(5, 512), np.arange(5))
    assert cache.exists("test_key") is True


def test_load_missing_key_raises(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No cached embeddings"):
        cache.load("nonexistent_key")


def test_save_rejects_mismatched_sample_counts(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    embeddings = np.random.rand(10, 512)
    labels = np.arange(5)  # wrong count
    with pytest.raises(ValueError, match="Mismatched sample counts"):
        cache.save("test_key", embeddings, labels)