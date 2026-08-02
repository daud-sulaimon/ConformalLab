"""
DatasetManager: factory for constructing ConformalLab datasets by name.

Provides a single entry point, `DatasetManager(name, **kwargs)`, so the
rest of the framework never needs to know or care which concrete
dataset class backs a given name. Internally uses a name -> class
lookup table rather than an if/elif chain, keeping extension to new
datasets a one-line addition to `_DATASET_REGISTRY`.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from src.datasets.base import BaseDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Populated as each concrete dataset is implemented. Deliberately a
# plain dict, not a decorator-based registry: with a small, fixed set
# of datasets known in advance, this stays simple to read top-to-bottom
# and matches the Factory pattern described in the project's
# architecture documentation.
_DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


class DatasetManager:
    """
    Factory that constructs the correct dataset implementation by name.

    Parameters
    ----------
    name
        Dataset identifier. Must be a key in `_DATASET_REGISTRY`
        (e.g. ``"imagenet"``).
    **kwargs
        Forwarded to the underlying dataset class's constructor
        (e.g. ``root``, ``batch_size``, ``num_workers``).

    Raises
    ------
    ValueError
        If `name` is not a recognised dataset.

    Examples
    --------
    >>> dataset = DatasetManager("imagenet", root="data/imagenet")
    >>> dataset.load()
    >>> calibration = dataset.calibration_loader()
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        if name not in _DATASET_REGISTRY:
            available = sorted(_DATASET_REGISTRY.keys())
            raise ValueError(
                f"Unknown dataset '{name}'. Available datasets: {available}"
            )

        dataset_class = _DATASET_REGISTRY[name]
        logger.info(f"Constructing dataset '{name}' ({dataset_class.__name__})")
        self._dataset: BaseDataset = dataset_class(**kwargs)

    def __getattr__(self, item: str) -> Any:
        # Delegate everything else (load, calibration_loader, etc.) to
        # the wrapped concrete dataset instance, so DatasetManager
        # transparently exposes the full BaseDataset interface.
        return getattr(self._dataset, item)