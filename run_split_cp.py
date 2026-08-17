"""
"""
Run a chosen conformal method (LAC/APS/RAPS) on cached EXP001 embeddings.

Loads the calibration and test embeddings already extracted and
cached by run.py (EXP001), reconstructs class probabilities using
CLIP's text embeddings and logit scale, calibrates the chosen method,
reports empirical coverage against the target from
configs/default.yaml, and freezes the calibration threshold (q_hat)
to disk so it can be reused unchanged by run_shift_eval.py on
distribution-shift datasets.

Usage:
    python run_split_cp.py --config configs/default.yaml --method lac
    python run_split_cp.py --config configs/default.yaml --method aps
    python run_split_cp.py --config configs/default.yaml --method raps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import src.datasets.imagenet  # noqa: F401  (registers ImageNetDataset)
import src.models.clip_model  # noqa: F401  (registers CLIPModel)
from src.conformal.factory import create_conformal_method
from src.datasets.imagenet_class_names import load_imagenet_class_names
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
    parser = argparse.ArgumentParser(
        description="Run a conformal method on cached EXP001 embeddings."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--method", type=str, required=True, choices=["lac", "aps", "raps"]
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    set_seed(config.seed.value)

    model = ModelManager(config.model.name)
    model.load()

    class_names = load_imagenet_class_names()
    text_embeddings = model.encode_text(class_names).cpu().numpy()

    cache = EmbeddingCache()
    cache_key_prefix = f"{config.model.name}_{config.dataset.name}"

    calibration_embeddings, calibration_labels = cache.load(f"{cache_key_prefix}_calibration")
    test_embeddings, test_labels = cache.load(f"{cache_key_prefix}_test")

    calibration_probs = _softmax(model.logit_scale * (calibration_embeddings @ text_embeddings.T))
    test_probs = _softmax(model.logit_scale * (test_embeddings @ text_embeddings.T))

    method = create_conformal_method(args.method, alpha=config.calibration.alpha)
    method.calibrate(calibration_probs, calibration_labels)
    prediction_sets = method.predict_sets(test_probs)

    report = coverage_report(prediction_sets, test_labels, alpha=config.calibration.alpha)

    print(f"\n--- {args.method.upper()} Coverage Report ---")
    for key, value in report.items():
        print(f"{key}: {value}")

    output_dir = Path("results") / f"{config.experiment.id}-{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "coverage.json", "w", encoding="utf-8") as f:
        json.dump({"method": args.method, **report}, f, indent=2)

    threshold_path = output_dir / "threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"alpha": config.calibration.alpha, "q_hat": method.q_hat}, f, indent=2)

    logger.info(f"Coverage report archived to {output_dir / 'coverage.json'}")
    logger.info(f"Calibration threshold frozen and archived to {threshold_path}")


if __name__ == "__main__":
    main()