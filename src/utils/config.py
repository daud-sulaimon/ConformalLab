"""
Configuration loading and validation for ConformalLab.

Every experiment is defined by a single YAML file. This module loads
that YAML into strongly-typed, immutable Python objects and validates
every field before any dataset, model, or GPU work begins — so a
malformed config fails in milliseconds with a clear error message,
rather than after an hour of wasted embedding extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_MODELS = {"clip", "siglip"}
_VALID_DATASETS = {"imagenet", "imagenet_r", "imagenet_v2", "imagenet_a"}


class ConfigError(ValueError):
    """Raised when a configuration file is missing fields or has invalid values."""


@dataclass(frozen=True)
class ExperimentConfig:
    id: str


@dataclass(frozen=True)
class ModelConfig:
    name: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    subset_size: int = 1000


@dataclass(frozen=True)
class CalibrationConfig:
    alpha: float


@dataclass(frozen=True)
class SeedConfig:
    value: int


@dataclass(frozen=True)
class Config:
    """Fully validated, immutable representation of one experiment config."""

    experiment: ExperimentConfig
    model: ModelConfig
    dataset: DatasetConfig
    calibration: CalibrationConfig
    seed: SeedConfig


def _require_keys(section: Dict[str, Any], keys: List[str], section_name: str) -> None:
    """Raise ConfigError listing every missing key, not just the first one found."""
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(
            f"Missing required field(s) {missing} in '{section_name}' section."
        )


def load_config(path: Union[str, Path]) -> Config:
    """
    Load and validate an experiment configuration from a YAML file.

    Parameters
    ----------
    path
        Path to a YAML configuration file (e.g. ``configs/default.yaml``).

    Returns
    -------
    Config
        Fully validated, immutable configuration object.

    Raises
    ------
    ConfigError
        If the file is missing, unparseable, missing required fields,
        or contains invalid values (e.g. an unrecognised model name,
        or an alpha outside (0, 1)).

    Examples
    --------
    >>> config = load_config("configs/default.yaml")
    >>> config.model.name
    'clip'
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            raw: Any = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ConfigError(f"Failed to parse YAML in {path}: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration file {path} must contain a YAML mapping.")

    _require_keys(raw, ["experiment", "model", "dataset", "calibration", "seed"], "top-level")
    _require_keys(raw["experiment"], ["id"], "experiment")
    _require_keys(raw["model"], ["name"], "model")
    _require_keys(raw["dataset"], ["name"], "dataset")
    _require_keys(raw["calibration"], ["alpha"], "calibration")
    _require_keys(raw["seed"], ["value"], "seed")

    model_name = raw["model"]["name"]
    if model_name not in _VALID_MODELS:
        raise ConfigError(
            f"Unknown model '{model_name}'. Must be one of {sorted(_VALID_MODELS)}."
        )

    dataset_name = raw["dataset"]["name"]
    if dataset_name not in _VALID_DATASETS:
        raise ConfigError(
            f"Unknown dataset '{dataset_name}'. Must be one of {sorted(_VALID_DATASETS)}."
        )

    subset_size = raw["dataset"].get("subset_size", 1000)
    if not isinstance(subset_size, int) or isinstance(subset_size, bool) or subset_size <= 0:
        raise ConfigError(
            f"'dataset.subset_size' must be a positive integer, got {subset_size!r}."
        )

    alpha = raw["calibration"]["alpha"]
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not (0 < alpha < 1):
        raise ConfigError(
            f"'calibration.alpha' must be a number strictly between 0 and 1, got {alpha!r}."
        )

    seed_value = raw["seed"]["value"]
    if not isinstance(seed_value, int) or isinstance(seed_value, bool):
        raise ConfigError(f"'seed.value' must be an integer, got {seed_value!r}.")

    config = Config(
        experiment=ExperimentConfig(id=str(raw["experiment"]["id"])),
        model=ModelConfig(name=model_name),
        dataset=DatasetConfig(name=dataset_name, subset_size=subset_size),
        calibration=CalibrationConfig(alpha=float(alpha)),
        seed=SeedConfig(value=seed_value),
    )

    logger.info(f"Loaded configuration '{config.experiment.id}' from {path}")
    return config