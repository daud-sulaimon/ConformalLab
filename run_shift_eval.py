"""
Evaluate a conformal method's coverage on a distribution-shift dataset,
using a threshold already frozen by run_split_cp.py on ImageNet for
that same method. This script never recalibrates — it deliberately
reuses the exact q_hat computed on unshifted ImageNet, so that any
coverage drop observed reflects the shift itself, not a different
calibration.

Text embeddings (and therefore the softmax probability space) are
built from the shift dataset's OWN class_names() — not always the
full 1000-class ImageNet list — since some shift datasets (ImageNet-R,
ImageNet-A) restrict evaluation to a subset of classes.

Usage:
    python run_shift_eval.py --config configs/default.yaml --dataset imagenet_v2 --method aps
    python run_shift_eval.py --config configs/default.yaml --dataset imagenet_r --method raps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import src.datasets.imagenet  # noqa: F401  (registers ImageNetDataset)
import src.datasets.imagenet_a  # noqa: F401  (registers ImageNetADataset)
import src.datasets.imagenet_r  # noqa: F401  (registers ImageNetRDataset)
import src.datasets.imagenet_v2  # noqa: F401  (registers ImageNetV2Dataset)
import src.models.clip_model  # noqa: F401  (registers CLIPModel)
from src.conformal.factory import create_conformal_method
from src.datasets.imagenet_class_names import load_imagenet_class_names
from src.datasets.manager import DatasetManager
from src.embeddings.cache import EmbeddingCache
from src.metrics.coverage import coverage_report
from src.models.manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

_SHIFT_EXPERIMENT_IDS = {
    "imagenet_v2": "EXP002",
    "imagenet_r": "EXP003",
    "imagenet_a": "EXP004",
}


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a conformal method's coverage on a shift dataset using a frozen threshold."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, required=True, choices=sorted(_SHIFT_EXPERIMENT_IDS.keys())
    )
    parser.add_argument(
        "--method", type=str, required=True, choices=["lac", "aps", "raps"]
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    set_seed(config.seed.value)

    if config.dataset.name != "imagenet":
        raise ValueError(
            f"run_shift_eval.py requires a config calibrated on 'imagenet' "
            f"(the frozen baseline), got '{config.dataset.name}'."
        )

    baseline_dir = Path("results") / f"{config.experiment.id}-{args.method}"
    threshold_path = baseline_dir / "threshold.json"
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"No frozen threshold found at {threshold_path}. "
            f"Run 'python run_split_cp.py --config {args.config} --method {args.method}' first."
        )

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    q_hat = threshold_data["q_hat"]
    alpha = threshold_data["alpha"]

    logger.info(f"Loaded frozen threshold ({args.method}): alpha={alpha}, q_hat={q_hat:.4f}")

    model = ModelManager(config.model.name)
    model.load()

    full_class_names = load_imagenet_class_names()

    shift_dataset = DatasetManager(
        args.dataset,
        class_names=full_class_names,
        subset_size=config.dataset.subset_size,
        transform=model.preprocess,
    )
    shift_dataset.load()

    active_class_names = shift_dataset.class_names()
    text_embeddings = model.encode_text(active_class_names).cpu().numpy()

    cache = EmbeddingCache()
    cache_key = f"{config.model.name}_{args.dataset}_test"

    if cache.exists(cache_key):
        logger.info(f"Using cached embeddings for '{cache_key}'.")
        shift_embeddings, shift_labels = cache.load(cache_key)
    else:
        from src.embeddings.extractor import _extract_from_loader

        loader = shift_dataset.test_loader()
        shift_embeddings, shift_labels = _extract_from_loader(loader, model)
        cache.save(cache_key, shift_embeddings, shift_labels)

    shift_probs = _softmax(model.logit_scale * (shift_embeddings @ text_embeddings.T))

    # Reconstruct the method and apply the frozen threshold directly,
    # without recalibrating - this is what "frozen" means.
    method = create_conformal_method(args.method, alpha=alpha)
    method._q_hat = q_hat  # bypass calibrate(): we are deliberately reusing a frozen threshold
    prediction_sets = method.predict_sets(shift_probs)

    report = coverage_report(prediction_sets, shift_labels, alpha=alpha)
    report["num_classes_evaluated"] = len(active_class_names)

    print(f"\n--- {args.method.upper()} Shift Evaluation: {args.dataset} ---")
    for key, value in report.items():
        print(f"{key}: {value}")

    experiment_id = _SHIFT_EXPERIMENT_IDS[args.dataset]
    output_dir = Path("results") / f"{experiment_id}-{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "coverage.json", "w", encoding="utf-8") as f:
        json.dump({"method": args.method, "dataset": args.dataset, **report}, f, indent=2)

    logger.info(f"Coverage report archived to {output_dir / 'coverage.json'}")


if __name__ == "__main__":
    main()