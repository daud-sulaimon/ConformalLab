"""
CLIP wrapper for ConformalLab.

Hides OpenCLIP's API completely behind the BaseModel interface, so
the rest of the framework never imports `open_clip` directly. If CLIP
is later swapped for SigLIP, only this file (and a corresponding
siglip_model.py) changes — nothing downstream.

Implements CLIP's standard zero-shot classification approach: encode
each candidate class name as a text prompt, encode the image, and
score the image against every class via cosine similarity + softmax.
Text embeddings for a given class list are cached, since they never
change between calls for the same dataset.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import open_clip
import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_NAME = "ViT-B-32-quickgelu"
_DEFAULT_PRETRAINED = "openai"
_DEFAULT_PROMPT_TEMPLATE = "a photo of a {}."


class CLIPModel(BaseModel):
    """
    CLIP zero-shot classifier, wrapping OpenCLIP.

    Parameters
    ----------
    model_name
        OpenCLIP model architecture identifier. Defaults to
        ``"ViT-B-32"``, matching the project's EXP001 specification.
    pretrained
        Which pretrained weights to load. Defaults to ``"openai"``
        (the original CLIP release), for comparability with prior
        published CLIP results.
    prompt_template
        Template string used to turn a class name into a text prompt.
        Must contain exactly one ``{}`` placeholder. Defaults to
        ``"a photo of a {}."``, following the original CLIP paper.
    device
        Torch device to run on. If ``None``, automatically selects
        ``"cuda"`` if available, otherwise ``"cpu"``.

    Examples
    --------
    >>> model = CLIPModel()
    >>> model.load()
    >>> probs = model.predict(images, class_names=["cat", "dog"])
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        pretrained: str = _DEFAULT_PRETRAINED,
        prompt_template: str = _DEFAULT_PROMPT_TEMPLATE,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._prompt_template = prompt_template
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[torch.nn.Module] = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False

        # Text embedding cache: avoids recomputing embeddings for the
        # same fixed class list (e.g. ImageNet's 1000 classes) on
        # every predict() call.
        self._cached_class_names: Optional[Tuple[str, ...]] = None
        self._cached_text_embeddings: Optional[torch.Tensor] = None

    def load(self) -> None:
        """Load CLIP weights and move the model to the target device."""
        logger.info(
            f"Loading CLIP ({self._model_name}, pretrained={self._pretrained}) "
            f"on device '{self._device}'..."
        )

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)

        self._model.to(self._device)
        self._model.eval()
        self._loaded = True

        logger.info("CLIP loaded successfully.")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "CLIPModel.load() must be called before encode_images(), "
                "encode_text(), or predict()."
            )

    @property
    def preprocess(self):
        """
        CLIP's official image preprocessing transform (resize, crop,
        normalize). Pass this to a dataset's `transform` argument so
        images are prepared exactly as CLIP expects.
        """
        self._require_loaded()
        return self._preprocess

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        self._require_loaded()
        images = images.to(self._device)
        embeddings = self._model.encode_image(images)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        """
        Encode class names into text embeddings.

        Results are cached: if `class_names` matches the most recently
        encoded list, the cached embeddings are returned instead of
        recomputing them.
        """
        self._require_loaded()

        class_names_key = tuple(class_names)
        if class_names_key == self._cached_class_names:
            return self._cached_text_embeddings

        prompts = [self._prompt_template.format(name) for name in class_names]
        tokens = self._tokenizer(prompts).to(self._device)
        embeddings = F.normalize(self._model.encode_text(tokens), dim=-1)

        self._cached_class_names = class_names_key
        self._cached_text_embeddings = embeddings
        return embeddings

    @torch.no_grad()
    def predict(self, images: torch.Tensor, class_names: List[str]) -> torch.Tensor:
        """
        Produce class probabilities via CLIP's zero-shot approach:
        cosine similarity between image and text embeddings, scaled by
        CLIP's learned temperature, then softmax.
        """
        self._require_loaded()
        image_embeddings = self.encode_images(images)
        text_embeddings = self.encode_text(class_names)

        logit_scale = self._model.logit_scale.exp()
        similarity = logit_scale * image_embeddings @ text_embeddings.T
        return similarity.softmax(dim=-1)