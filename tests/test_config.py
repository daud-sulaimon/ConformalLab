"""Tests for src.utils.config."""

import pytest

from src.utils.config import ConfigError, load_config

VALID_YAML = """
experiment:
  id: EXP001

model:
  name: clip

dataset:
  name: imagenet

calibration:
  alpha: 0.1

seed:
  value: 42
"""


def _write(tmp_path, content: str, filename: str = "config.yaml"):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_load_valid_config(tmp_path):
    path = _write(tmp_path, VALID_YAML)
    config = load_config(path)

    assert config.experiment.id == "EXP001"
    assert config.model.name == "clip"
    assert config.dataset.name == "imagenet"
    assert config.calibration.alpha == 0.1
    assert config.seed.value == 42


def test_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load_config(missing_path)


def test_missing_top_level_section_raises(tmp_path):
    bad_yaml = """
model:
  name: clip
dataset:
  name: imagenet
calibration:
  alpha: 0.1
seed:
  value: 42
"""
    path = _write(tmp_path, bad_yaml)
    with pytest.raises(ConfigError, match="experiment"):
        load_config(path)


def test_unknown_model_raises(tmp_path):
    bad_yaml = VALID_YAML.replace("name: clip", "name: gpt4v")
    path = _write(tmp_path, bad_yaml)
    with pytest.raises(ConfigError, match="Unknown model"):
        load_config(path)


def test_invalid_alpha_raises(tmp_path):
    bad_yaml = VALID_YAML.replace("alpha: 0.1", "alpha: 1.5")
    path = _write(tmp_path, bad_yaml)
    with pytest.raises(ConfigError, match="alpha"):
        load_config(path)


def test_malformed_yaml_raises(tmp_path):
    path = _write(tmp_path, "experiment: [unclosed")
    with pytest.raises(ConfigError):
        load_config(path)


def test_dataset_subset_size_defaults_to_1000(tmp_path):
    path = _write(tmp_path, VALID_YAML)
    config = load_config(path)
    assert config.dataset.subset_size == 1000


def test_dataset_subset_size_can_be_overridden(tmp_path):
    custom_yaml = VALID_YAML.replace(
        "name: imagenet\n", "name: imagenet\n  subset_size: 500\n"
    )
    path = _write(tmp_path, custom_yaml)
    config = load_config(path)
    assert config.dataset.subset_size == 500