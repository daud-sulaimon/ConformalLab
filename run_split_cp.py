"""
Run Split Conformal Prediction on cached EXP001 embeddings.

Loads the calibration and test embeddings already extracted and
cached by run.py (EXP001), reconstructs class probabilities using
CLIP's text embeddings and logit scale, calibrates Split CP, and
reports empirical coverage against the target from configs/default.yaml.

Usage:
    python run_split_cp.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

import src.datasets.imagenet  # noqa: F401  (registers ImageNetDataset)
import src.models.clip_model  # noqa: F401  (registers CLIPModel)
from src.conformal.split_cp import SplitConformalMethod
from src.datasets.manager import DatasetManager
from src.embeddings.cache import EmbeddingCache
from src.metrics.coverage import coverage_report
from src.models.manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Split CP on cached EXP001 embeddings.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    set_seed(config.seed.value)

    model = ModelManager(config.model.name)
    model.load()

    class_names = DatasetManager(
        config.dataset.name, subset_size=1, transform=model.preprocess
    )
    # Only need class_names here, not real images - a tiny subset_size
    # avoids re-streaming the full dataset just to read the label list.
    class_names.load()
    class_names = class_names.class_names()

    text_embeddings = model.encode_text(class_names).cpu().numpy()

    cache = EmbeddingCache()
    cache_key_prefix = f"{config.model.name}_{config.dataset.name}"

    calibration_embeddings, calibration_labels = cache.load(f"{cache_key_prefix}_calibration")
    test_embeddings, test_labels = cache.load(f"{cache_key_prefix}_test")

    calibration_probs = _softmax(model.logit_scale * (calibration_embeddings @ text_embeddings.T))
    test_probs = _softmax(model.logit_scale * (test_embeddings @ text_embeddings.T))

    method = SplitConformalMethod(alpha=config.calibration.alpha)
    method.calibrate(calibration_probs, calibration_labels)
    prediction_sets = method.predict_sets(test_probs)

    report = coverage_report(prediction_sets, test_labels, alpha=config.calibration.alpha)

    print("\n--- Split CP Coverage Report ---")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()