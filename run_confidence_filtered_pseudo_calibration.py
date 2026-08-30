"""
C4: Confidence-filtered hard pseudo-calibration for zero-shot CLIP.

Motivated by uncertainty-aware pseudo-calibration literature
(Siahkali et al., 2026), which mitigates pseudo-calibration failure by
weighting/interpolating pseudo-labels by model confidence. This script
implements a simpler operating-point variant - hard filtering at fixed
confidence thresholds - as an exploratory study of the same underlying
question, NOT a reproduction of Siahkali et al.'s full source-tuned
procedure. That distinction is deliberate and should be preserved in
any write-up.

Research question: does the gain in pseudo-label precision obtained by
confidence filtering translate into improved conformal calibration, or
is it offset by the loss of representative target observations and
the resulting reduction in effective calibration size? Raising the
threshold simultaneously increases pseudo-label precision AND shrinks
the effective calibration sample - both effects are measured and
reported separately (mean_pseudo_label_precision vs
mean_effective_calibration_n), so any observed coverage change can be
attributed rather than assumed.

The four thresholds {0.0, 0.5, 0.7, 0.9} are predefined exploratory
operating points, fixed before results are examined - not a search for
an "optimal" threshold, and none should be reported as such.

No label leakage: the confidence filter (keep_mask) depends only on
the model's own top-1 probability, computed without target labels.
True labels appear only in the true-label oracle control (isolating
selection effect from pseudo-label effect) and in final evaluation.

Design: for each dataset, method, and threshold - filter the fixed
250-example calibration pool by confidence, then calibrate TWICE on
the identical filtered sample: once with pseudo-labels, once with true
labels (the oracle control). Both are evaluated against true labels on
the same held-out set. The difference between the two conditions
isolates pseudo-label error from the sample-selection effect of
filtering itself.

Reads cached embeddings only. No new inference.

Usage:
    python run_confidence_filtered_pseudo_calibration.py
"""

from __future__ import annotations

import json
import logging
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
CONFIDENCE_THRESHOLDS = [0.0, 0.5, 0.7, 0.9]  # 0.0 = unfiltered, matches C3


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _load_probs_and_labels(dataset: str, model, full_class_names, cache):
    """Reconstruct probabilities in the correct class space for this dataset.

    Note: the 'imagenet' condition combines the calibration and test
    caches into one 1000-example same-distribution reference pool, then
    randomly partitions it for calibration/evaluation each draw. This
    is a same-distribution reference condition, not a genuine
    target-shift experiment - reported and interpreted as such.
    """
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
        f"{'dataset':>13} {'method':>6} {'thresh':>7} {'acc':>7} "
        f"{'N_eff':>7} {'pseudo_prec':>12} "
        f"{'pseudo_cov':>11} {'true_cov':>9} {'gap':>8} {'set_size':>9}"
    ]

    for dataset in DATASETS:
        probs, labels, n_classes = _load_probs_and_labels(
            dataset, model, full_class_names, cache
        )
        pseudo_labels = probs.argmax(axis=1)
        top1_conf = probs.max(axis=1)
        accuracy = float((pseudo_labels == labels).mean())

        total = probs.shape[0]
        if total < CALIBRATION_N + EVAL_SIZE:
            logger.warning(f"{dataset}: {total} examples < required. Skipping.")
            continue

        for method in METHODS:
            for threshold in CONFIDENCE_THRESHOLDS:
                covs_pseudo, covs_true, sizes = [], [], []
                kept_fracs, precisions, effective_ns = [], [], []

                for draw in range(NUM_DRAWS):
                    rng = np.random.default_rng(config.seed.value + draw)
                    perm = rng.permutation(total)
                    cal_idx = perm[:CALIBRATION_N]
                    eval_idx = perm[CALIBRATION_N:CALIBRATION_N + EVAL_SIZE]

                    cal_conf = top1_conf[cal_idx]
                    keep_mask = cal_conf >= threshold
                    n_kept = int(keep_mask.sum())
                    kept_fracs.append(float(keep_mask.mean()))
                    effective_ns.append(n_kept)

                    if n_kept < 5:
                        # Too few examples survive filtering to calibrate meaningfully.
                        continue

                    filtered_cal_idx = cal_idx[keep_mask]
                    filtered_pseudo = pseudo_labels[filtered_cal_idx]
                    filtered_true = labels[filtered_cal_idx]
                    precisions.append(float((filtered_pseudo == filtered_true).mean()))

                    cal_probs = probs[filtered_cal_idx]
                    eval_probs, eval_labels = probs[eval_idx], labels[eval_idx]

                    kwargs = {"alpha": alpha}
                    if method in ("aps", "raps"):
                        kwargs["randomize"] = True
                        kwargs["seed"] = config.seed.value + draw

                    m_pseudo = create_conformal_method(method, **kwargs)
                    m_pseudo.calibrate(cal_probs, filtered_pseudo)
                    r_pseudo = coverage_report(
                        m_pseudo.predict_sets(eval_probs), eval_labels, alpha=alpha
                    )

                    m_true = create_conformal_method(method, **kwargs)
                    m_true.calibrate(cal_probs, filtered_true)
                    r_true = coverage_report(
                        m_true.predict_sets(eval_probs), eval_labels, alpha=alpha
                    )

                    covs_pseudo.append(r_pseudo["empirical_coverage"])
                    covs_true.append(r_true["empirical_coverage"])
                    sizes.append(r_pseudo["average_set_size"])

                if not covs_pseudo:
                    logger.warning(
                        f"{dataset}/{method}/thresh={threshold}: no valid draws "
                        f"(filter too strict). Skipping."
                    )
                    continue

                row = {
                    "dataset": dataset,
                    "method": method,
                    "threshold": threshold,
                    "top1_accuracy": accuracy,
                    "mean_kept_fraction": float(np.mean(kept_fracs)),
                    "mean_effective_calibration_n": float(np.mean(effective_ns)),
                    "min_effective_calibration_n": int(np.min(effective_ns)),
                    "max_effective_calibration_n": int(np.max(effective_ns)),
                    "mean_pseudo_label_precision": float(np.mean(precisions)),
                    "pseudo_coverage_mean": float(np.mean(covs_pseudo)),
                    "pseudo_coverage_std": float(np.std(covs_pseudo)),
                    "true_coverage_mean": float(np.mean(covs_true)),
                    "true_coverage_std": float(np.std(covs_true)),
                    "coverage_gap": float(np.mean(covs_pseudo) - np.mean(covs_true)),
                    "set_size_mean": float(np.mean(sizes)),
                    "num_valid_draws": len(covs_pseudo),
                }
                results.append(row)

                line = (
                    f"{dataset:>13} {method:>6} {threshold:>7.1f} {accuracy:>7.3f} "
                    f"{row['mean_effective_calibration_n']:>7.1f} "
                    f"{row['mean_pseudo_label_precision']:>12.3f} "
                    f"{row['pseudo_coverage_mean']:>11.4f} {row['true_coverage_mean']:>9.4f} "
                    f"{row['coverage_gap']:>+8.4f} {row['set_size_mean']:>9.2f}"
                )
                lines.append(line)
                print(line)

    out_dir = Path("results/pseudo_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "c4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "c4_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWritten to {out_dir}/c4_results.txt")


if __name__ == "__main__":
    main()