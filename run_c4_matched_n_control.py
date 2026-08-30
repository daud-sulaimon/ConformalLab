"""
C4-control: matched-effective-N random selection vs confidence-filtered
selection, holding N and label quality (true labels) fixed.

Isolates the SELECTION effect from the SAMPLE-SIZE effect in C4. For
each (dataset, method, threshold) combination already run in C4, reads
that combination's mean effective calibration size N_eff, then draws
N_eff RANDOM calibration examples (no confidence filtering) using true
labels, and compares against C4's true-label oracle coverage at that
same threshold. Both arms use true labels throughout - pseudo-labels
are deliberately excluded here, since the question is purely "does
WHICH examples matter, holding label quality fixed."

Requires results/pseudo_calibration/c4_results.json to exist (run
run_confidence_filtered_pseudo_calibration.py first).

Reads cached embeddings only. No new inference.

Usage:
    python run_c4_matched_n_control.py
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

NUM_DRAWS = 20
EVAL_SIZE = 500
CALIBRATION_N = 250  # matches the pool size C4 filtered from


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _load_probs_and_labels(dataset: str, model, full_class_names, cache):
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

    c4_path = Path("results/pseudo_calibration/c4_results.json")
    if not c4_path.exists():
        raise FileNotFoundError(
            f"{c4_path} not found. Run run_confidence_filtered_pseudo_calibration.py first."
        )
    with open(c4_path, "r", encoding="utf-8") as f:
        c4_results = json.load(f)

    config = load_config("configs/default.yaml")
    set_seed(config.seed.value)
    alpha = config.calibration.alpha

    model = ModelManager("clip")
    model.load()
    full_class_names = load_imagenet_class_names()
    cache = EmbeddingCache()

    # Cache loaded probs/labels per dataset once, reused across methods/thresholds.
    dataset_cache: dict = {}

    results = []
    lines = [
        f"{'dataset':>13} {'method':>6} {'thresh':>7} {'N_matched':>9} "
        f"{'filtered_true_cov':>18} {'random_true_cov':>16} {'selection_effect':>17}"
    ]

    for row in c4_results:
        dataset, method, threshold = row["dataset"], row["method"], row["threshold"]
        if threshold == 0.0:
            continue  # threshold 0.0 IS the random/unfiltered condition already

        n_matched = int(round(row["mean_effective_calibration_n"]))
        if n_matched < 5:
            logger.warning(f"{dataset}/{method}/{threshold}: N_matched < 5, skipping.")
            continue

        if dataset not in dataset_cache:
            dataset_cache[dataset] = _load_probs_and_labels(
                dataset, model, full_class_names, cache
            )
        probs, labels, n_classes = dataset_cache[dataset]
        total = probs.shape[0]

        random_true_covs = []
        for draw in range(NUM_DRAWS):
            rng = np.random.default_rng(config.seed.value + draw)
            perm = rng.permutation(total)
            cal_idx_full = perm[:CALIBRATION_N]
            eval_idx = perm[CALIBRATION_N:CALIBRATION_N + EVAL_SIZE]

            # Random N_matched examples drawn from the SAME calibration
            # pool C4 filtered from (not the full dataset), so the only
            # difference from C4's filtered arm is the selection rule.
            random_sub_rng = np.random.default_rng(config.seed.value + draw + 900_000)
            random_idx = random_sub_rng.choice(cal_idx_full, size=n_matched, replace=False)

            cal_probs = probs[random_idx]
            cal_true_labels = labels[random_idx]
            eval_probs, eval_labels = probs[eval_idx], labels[eval_idx]

            kwargs = {"alpha": alpha}
            if method in ("aps", "raps"):
                kwargs["randomize"] = True
                kwargs["seed"] = config.seed.value + draw

            m = create_conformal_method(method, **kwargs)
            m.calibrate(cal_probs, cal_true_labels)
            r = coverage_report(m.predict_sets(eval_probs), eval_labels, alpha=alpha)
            random_true_covs.append(r["empirical_coverage"])

        random_true_cov_mean = float(np.mean(random_true_covs))
        filtered_true_cov = row["true_coverage_mean"]
        selection_effect = filtered_true_cov - random_true_cov_mean

        out_row = {
            "dataset": dataset,
            "method": method,
            "threshold": threshold,
            "n_matched": n_matched,
            "filtered_true_coverage": filtered_true_cov,
            "random_true_coverage_mean": random_true_cov_mean,
            "random_true_coverage_std": float(np.std(random_true_covs)),
            "selection_effect": selection_effect,
        }
        results.append(out_row)

        line = (
            f"{dataset:>13} {method:>6} {threshold:>7.1f} {n_matched:>9} "
            f"{filtered_true_cov:>18.4f} {random_true_cov_mean:>16.4f} "
            f"{selection_effect:>+17.4f}"
        )
        lines.append(line)
        print(line)

    out_dir = Path("results/pseudo_calibration")
    with open(out_dir / "c4_control_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "c4_control_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWritten to {out_dir}/c4_control_results.txt")


if __name__ == "__main__":
    main()