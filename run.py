"""
Experiment runner for ConformalLab.

Single entry point tying together configuration loading, deterministic
seeding, dataset loading, model loading, and embedding extraction:

    python run.py --config configs/default.yaml

This is EXP001's execution harness. It validates that the full
pipeline (config -> dataset -> model -> embeddings -> metrics) works
end to end, and archives results into results/<experiment_id>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

# Importing these registers ImageNetDataset and CLIPModel with their
# respective managers (see the bottom of imagenet.py / clip_model.py).
import src.datasets.imagenet  # noqa: F401
import src.models.clip_model  # noqa: F401
from src.datasets.manager import DatasetManager
from src.embeddings.cache import EmbeddingCache
from src.embeddings.extractor import extract_embeddings
from src.models.manager import ModelManager
from src.utils.config import Config, load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)



def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Compute top-k accuracy: the fraction of samples where the true
    label appears among the k highest-probability predicted classes.
    """
    top_k_preds = np.argsort(-probs, axis=1)[:, :k]
    correct = np.any(top_k_preds == labels[:, None], axis=1)
    return float(correct.mean())


def run_experiment(config: Config) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    """
    Execute one experiment end to end.

    Returns
    -------
    tuple
        ``(metrics, probs, test_labels, class_names)``, where `probs`
        and `test_labels` are used afterward to write predictions.csv.
    """
    set_seed(config.seed.value)
    logger.info(f"Starting experiment '{config.experiment.id}'")
    start_time = time.time()

    model = ModelManager(config.model.name)
    model.load()

    dataset = DatasetManager(
        config.dataset.name,
        subset_size=config.dataset.subset_size,
        transform=model.preprocess,
    )
    dataset.load()
    class_names = dataset.class_names()

    cache = EmbeddingCache()
    cache_key_prefix = f"{config.model.name}_{config.dataset.name}"

    embedding_start = time.time()
    extraction_result = extract_embeddings(
        dataset, model, cache_key_prefix=cache_key_prefix, cache=cache
    )
    embedding_elapsed = time.time() - embedding_start

    test_embeddings, test_labels = extraction_result["test"]
    num_test_images = test_embeddings.shape[0]

    # Score cached image embeddings against text embeddings directly,
    # rather than re-running images through the model (they aren't
    # stored, only their embeddings are).
    text_embeddings = model.encode_text(class_names).cpu().numpy()
    similarity = model.logit_scale * (test_embeddings @ text_embeddings.T)
    probs = _softmax(similarity)

    top1 = _top_k_accuracy(probs, test_labels, k=1)
    top5 = _top_k_accuracy(probs, test_labels, k=5)
    total_elapsed = time.time() - start_time

    metrics = {
        "experiment_id": config.experiment.id,
        "model": config.model.name,
        "dataset": config.dataset.name,
        "num_test_images": num_test_images,
        "embedding_dimension": test_embeddings.shape[1],
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "mean_inference_time_per_image_seconds": embedding_elapsed / num_test_images,
        "total_runtime_seconds": total_elapsed,
    }

    logger.info(
        f"Experiment '{config.experiment.id}' complete: "
        f"top1={top1:.4f}, top5={top5:.4f}, runtime={total_elapsed:.1f}s"
    )

    return metrics, probs, test_labels, class_names


def _write_predictions_csv(
    path: Path, probs: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> None:
    """Write one row per test image: true/predicted class and confidence."""
    predicted_indices = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_index", "true_label", "predicted_label", "confidence", "correct"]
        )
        for i, (true_idx, pred_idx, confidence) in enumerate(
            zip(labels, predicted_indices, confidences)
        ):
            writer.writerow(
                [
                    i,
                    class_names[true_idx],
                    class_names[pred_idx],
                    f"{confidence:.4f}",
                    true_idx == pred_idx,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ConformalLab experiment.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment YAML config."
    )
    args = parser.parse_args()

    log_file = configure_logging()

    config = load_config(args.config)
    metrics, probs, test_labels, class_names = run_experiment(config)

    output_dir = Path("results") / config.experiment.id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _write_predictions_csv(output_dir / "predictions.csv", probs, test_labels, class_names)

    shutil.copy(args.config, output_dir / "config.yaml")

    if log_file is not None:
        shutil.copy(log_file, output_dir / "runtime.log")

    logger.info(f"All outputs archived to {output_dir}")


if __name__ == "__main__":
    main()