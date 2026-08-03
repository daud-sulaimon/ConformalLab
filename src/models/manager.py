"""
ModelManager: factory for constructing ConformalLab models by name.

Provides a single entry point, `ModelManager(name, **kwargs)`, so the
rest of the framework never needs to know or care which concrete
model class backs a given name. Mirrors DatasetManager's design: a
name -> class lookup table populated by each concrete model
self-registering on import, rather than an if/elif chain.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Populated as each concrete model is implemented (see the bottom of
# clip_model.py). Kept as a plain dict rather than a decorator-based
# registry, matching the Factory pattern used for DatasetManager.
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


class ModelManager:
    """
    Factory that constructs the correct model implementation by name.

    Parameters
    ----------
    name
        Model identifier. Must be a key in `_MODEL_REGISTRY`
        (e.g. ``"clip"``).
    **kwargs
        Forwarded to the underlying model class's constructor
        (e.g. ``model_name``, ``pretrained``, ``device``).

    Raises
    ------
    ValueError
        If `name` is not a recognised model.

    Examples
    --------
    >>> model = ModelManager("clip")
    >>> model.load()
    >>> probs = model.predict(images, class_names)
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        if name not in _MODEL_REGISTRY:
            available = sorted(_MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available models: {available}"
            )

        model_class = _MODEL_REGISTRY[name]
        logger.info(f"Constructing model '{name}' ({model_class.__name__})")
        self._model: BaseModel = model_class(**kwargs)

    def __getattr__(self, item: str) -> Any:
        # Delegate everything else (load, encode_images, predict, etc.)
        # to the wrapped concrete model instance, so ModelManager
        # transparently exposes the full BaseModel interface.
        return getattr(self._model, item)