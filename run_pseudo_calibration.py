"""
C3: Pseudo-calibration for zero-shot CLIP under distribution shift.

Tests whether calibrating a conformal predictor on the model's OWN
predictions (pseudo-labels) instead of true labels yields usable
coverage, and how that degrades as target-domain accuracy falls.

Siahkali et al. (2026) bound target coverage by source classifier
loss, validated on MNIST/CIFAR with supervised models. Zero-shot CLIP
has substantially higher loss, and the four datasets here provide a
graded accuracy sweep. This script measures where pseudo-calibration
breaks.

Design: for each dataset and method, calibrate twice on the IDENTICAL
sample (same indices, same seed) - once with true labels, once with
pseudo-labels - and evaluate both against true labels on the same
held-out set. The only difference between conditions is the labels
used at calibration time.

ImageNet's calibration and test caches are combined into a single
1000-example pool, matching the sampling protocol already used for
V2/R/A (each cached as 1000 examples in one split).

Reads cached embeddings only. No new inference.

Usage:
    python run_pseudo_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.conformal.factory import create_conformal_method
from src.datasets.imagenet_class_names import load_imagenet_class_names
from src.embeddings.cache import EmbeddingCache
from src.metrics.coverage import coverage_report
from src.models.manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

import src.models.clip_model  # noqa: F401

logger = get_logger(__name__)

DATASETS = ["imagenet", "imagenet_v2", "imagenet_r", "imagenet_a"]
METHODS = ["lac", "aps", "raps"]
CALIBRATION_N = 250
NUM_DRAWS = 20
EVAL_SIZE = 500


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _load_probs_and_labels(dataset: str, model, full_class_names, cache):
    """Reconstruct probabilities in the correct class space for this dataset."""
    if dataset in ("imagenet_r", "imagenet_a"):
        if dataset == "imagenet_r":
            from src.datasets.imagenet_r_class_mapping import (
                load_imagenet_r_local_to_full_mapping as load_map,
            )
        else:
            from src.datasets.imagenet_a_class_mapping import (
                load_imagenet_a_local_to_full_mapping as load_map,
            )
        mapping = load_map()
        class_names = [full_class_names[mapping[i]] for i in range(len(mapping))]
        embeddings, labels = cache.load(f"clip_{dataset}_test")
    elif dataset == "imagenet_v2":
        class_names = full_class_names
        embeddings, labels = cache.load("clip_imagenet_v2_test")
    else:
        class_names = full_class_names
        cal_emb, cal_lab = cache.load("clip_imagenet_calibration")
        test_emb, test_lab = cache.load("clip_imagenet_test")
        embeddings = np.concatenate([cal_emb, test_emb], axis=0)
        labels = np.concatenate([cal_lab, test_lab], axis=0)

    text_emb = model.encode_text(class_names).cpu().numpy()
    probs = _softmax(model.logit_scale * (embeddings @ text_emb.T))
    return probs, labels, len(class_names)


def main() -> None:
    configure_logging()
    import logging
    logging.getLogger("conformallab").setLevel(logging.WARNING)

    config = load_config("configs/default.yaml")
    set_seed(config.seed.value)
    alpha = config.calibration.alpha

    model = ModelManager("clip")
    model.load()
    full_class_names = load_imagenet_class_names()
    cache = EmbeddingCache()

    results = []
    lines = [
        f"{'dataset':>13} {'method':>6} {'acc':>7} "
        f"{'true_cov':>9} {'pseudo_cov':>11} {'gap':>8} "
        f"{'true_size':>10} {'pseudo_size':>12}"
    ]

    for dataset in DATASETS:
        probs, labels, n_classes = _load_probs_and_labels(
            dataset, model, full_class_names, cache
        )
        pseudo_labels = probs.argmax(axis=1)
        accuracy = float((pseudo_labels == labels).mean())

        total = probs.shape[0]
        if total < CALIBRATION_N + EVAL_SIZE:
            logger.warning(
                f"{dataset}: only {total} cached examples; "
                f"need {CALIBRATION_N + EVAL_SIZE}. Skipping."
            )
            continue

        for method in METHODS:
            true_covs, pseudo_covs = [], []
            true_sizes, pseudo_sizes = [], []

            for draw in range(NUM_DRAWS):
                rng = np.random.default_rng(config.seed.value + draw)
                perm = rng.permutation(total)
                cal_idx = perm[:CALIBRATION_N]
                eval_idx = perm[CALIBRATION_N:CALIBRATION_N + EVAL_SIZE]

                cal_probs = probs[cal_idx]
                eval_probs, eval_labels = probs[eval_idx], labels[eval_idx]

                kwargs = {"alpha": alpha}
                if method in ("aps", "raps"):
                    kwargs["randomize"] = True
                    kwargs["seed"] = config.seed.value + draw

                # Condition A: true labels
                m_true = create_conformal_method(method, **kwargs)
                m_true.calibrate(cal_probs, labels[cal_idx])
                r_true = coverage_report(
                    m_true.predict_sets(eval_probs), eval_labels, alpha=alpha
                )

                # Condition B: pseudo-labels - identical sample, identical seed
                m_pseudo = create_conformal_method(method, **kwargs)
                m_pseudo.calibrate(cal_probs, pseudo_labels[cal_idx])
                r_pseudo = coverage_report(
                    m_pseudo.predict_sets(eval_probs), eval_labels, alpha=alpha
                )

                true_covs.append(r_true["empirical_coverage"])
                pseudo_covs.append(r_pseudo["empirical_coverage"])
                true_sizes.append(r_true["average_set_size"])
                pseudo_sizes.append(r_pseudo["average_set_size"])

            row = {
                "dataset": dataset,
                "method": method,
                "n_classes": n_classes,
                "top1_accuracy": accuracy,
                "true_coverage_mean": float(np.mean(true_covs)),
                "true_coverage_std": float(np.std(true_covs)),
                "pseudo_coverage_mean": float(np.mean(pseudo_covs)),
                "pseudo_coverage_std": float(np.std(pseudo_covs)),
                "coverage_gap": float(np.mean(pseudo_covs) - np.mean(true_covs)),
                "true_set_size_mean": float(np.mean(true_sizes)),
                "pseudo_set_size_mean": float(np.mean(pseudo_sizes)),
                "true_coverage_draws": [float(c) for c in true_covs],
                "pseudo_coverage_draws": [float(c) for c in pseudo_covs],
            }
            results.append(row)

            line = (
                f"{dataset:>13} {method:>6} {accuracy:>7.3f} "
                f"{row['true_coverage_mean']:>9.4f} {row['pseudo_coverage_mean']:>11.4f} "
                f"{row['coverage_gap']:>+8.4f} "
                f"{row['true_set_size_mean']:>10.2f} {row['pseudo_set_size_mean']:>12.2f}"
            )
            lines.append(line)
            print(line)

    out_dir = Path("results/pseudo_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "c3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "c3_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWritten to {out_dir}/c3_results.txt")


if __name__ == "__main__":
    main()