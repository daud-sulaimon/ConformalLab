"""Tests for src.models.clip_model. Downloads CLIP weights on first run
(cached afterward by open_clip) and requires network access."""

import pytest
import torch

from src.models.clip_model import CLIPModel


@pytest.fixture(scope="session")
def loaded_clip_model() -> CLIPModel:
    model = CLIPModel()
    model.load()
    return model


def test_clip_encode_images_produces_expected_shape(loaded_clip_model):
    dummy_images = torch.rand(2, 3, 224, 224)
    embeddings = loaded_clip_model.encode_images(dummy_images)
    assert embeddings.shape == (2, 512)  # ViT-B-32-quickgelu embedding dim


def test_clip_encode_text_produces_expected_shape(loaded_clip_model):
    embeddings = loaded_clip_model.encode_text(["cat", "dog", "car"])
    assert embeddings.shape == (3, 512)


def test_clip_encode_text_caches_repeated_class_list(loaded_clip_model):
    first = loaded_clip_model.encode_text(["cat", "dog"])
    second = loaded_clip_model.encode_text(["cat", "dog"])
    assert first is second  # same cached tensor object, not recomputed


def test_clip_predict_produces_valid_probability_distribution(loaded_clip_model):
    dummy_images = torch.rand(2, 3, 224, 224)
    probs = loaded_clip_model.predict(dummy_images, class_names=["cat", "dog"])

    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_calling_methods_before_load_raises():
    model = CLIPModel()
    with pytest.raises(RuntimeError, match="load"):
        model.encode_text(["cat"])