"""
Fourth quadrant of the 2x2 factorial design (selection x label-type):
random selection + pseudo-labels, at the SAME matched N as C4's
confidence-filtered condition.

Completes:
                    True labels        Pseudo labels
    Random          (matched-N ctrl)   (THIS SCRIPT)
    Filtered        (C4 true-label     (C4 pseudo-label
                      oracle)           condition)

The quantity this isolates:

    delta_filter_pseudo = filtered_pseudo_cov - random_pseudo_cov

at matched N - the actual effect of confidence filtering under
realistic (pseudo-labelled) calibration, separated from the
sample-size effect already isolated by the true-label matched-N
control.

Uses the SAME fixed N_matched per (dataset, method, threshold) as
run_c4_matched_n_control.py, documented there as an approximation
(mean effective N across draws, not each draw's own retained count) -
carried forward here for consistency between the two controls, since
comparing draw-exact N in one arm and mean N in the other would
introduce its own confound.

Requires results/pseudo_calibration/c4_results.json.
Reads cached embeddings only. No new inference.

Usage:
    python run_c4_random_pseudo.py
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
CALIBRATION_N = 250


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
        raise FileNotFoundError(f"{c4_path} not found. Run C4 first.")
    with open(c4_path, "r", encoding="utf-8") as f:
        c4_results = json.load(f)

    config = load_config("configs/default.yaml")
    set_seed(config.seed.value)
    alpha = config.calibration.alpha

    model = ModelManager("clip")
    model.load()
    full_class_names = load_imagenet_class_names()
    cache = EmbeddingCache()

    dataset_cache: dict = {}
    results = []
    lines = [
        f"{'dataset':>13} {'method':>6} {'thresh':>7} {'N':>5} "
        f"{'random_pseudo':>14} {'filtered_pseudo':>16} {'delta_filter':>13} "
        f"{'random_true':>12} {'selection_effect':>17}"
    ]

    for row in c4_results:
        dataset, method, threshold = row["dataset"], row["method"], row["threshold"]
        if threshold == 0.0:
            continue

        n_matched = int(round(row["mean_effective_calibration_n"]))
        if n_matched < 5:
            continue

        if dataset not in dataset_cache:
            dataset_cache[dataset] = _load_probs_and_labels(
                dataset, model, full_class_names, cache
            )
        probs, labels, n_classes = dataset_cache[dataset]
        pseudo_labels = probs.argmax(axis=1)
        total = probs.shape[0]

        random_pseudo_covs = []
        for draw in range(NUM_DRAWS):
            rng = np.random.default_rng(config.seed.value + draw)
            perm = rng.permutation(total)
            cal_idx_full = perm[:CALIBRATION_N]
            eval_idx = perm[CALIBRATION_N:CALIBRATION_N + EVAL_SIZE]

            # Identical random-subset construction to the true-label
            # control, for direct comparability - only the labels used
            # at calibration differ (pseudo vs true).
            random_sub_rng = np.random.default_rng(config.seed.value + draw + 900_000)
            random_idx = random_sub_rng.choice(cal_idx_full, size=n_matched, replace=False)

            cal_probs = probs[random_idx]
            cal_pseudo_labels = pseudo_labels[random_idx]
            eval_probs, eval_labels = probs[eval_idx], labels[eval_idx]

            kwargs = {"alpha": alpha}
            if method in ("aps", "raps"):
                kwargs["randomize"] = True
                kwargs["seed"] = config.seed.value + draw

            m = create_conformal_method(method, **kwargs)
            m.calibrate(cal_probs, cal_pseudo_labels)
            r = coverage_report(m.predict_sets(eval_probs), eval_labels, alpha=alpha)
            random_pseudo_covs.append(r["empirical_coverage"])

        random_pseudo_mean = float(np.mean(random_pseudo_covs))
        filtered_pseudo_cov = row["pseudo_coverage_mean"]
        delta_filter_pseudo = filtered_pseudo_cov - random_pseudo_mean

        out_row = {
            "dataset": dataset,
            "method": method,
            "threshold": threshold,
            "n_matched": n_matched,
            "random_pseudo_coverage_mean": random_pseudo_mean,
            "random_pseudo_coverage_std": float(np.std(random_pseudo_covs)),
            "filtered_pseudo_coverage": filtered_pseudo_cov,
            "delta_filter_pseudo": delta_filter_pseudo,
            "random_true_coverage": row.get("_random_true_coverage"),  # filled below if available
        }
        results.append(out_row)

        line = (
            f"{dataset:>13} {method:>6} {threshold:>7.1f} {n_matched:>5} "
            f"{random_pseudo_mean:>14.4f} {filtered_pseudo_cov:>16.4f} "
            f"{delta_filter_pseudo:>+13.4f} "
            f"{'--':>12} {'--':>17}"
        )
        lines.append(line)
        print(line)

    out_dir = Path("results/pseudo_calibration")
    with open(out_dir / "c4_random_pseudo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "c4_random_pseudo_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWritten to {out_dir}/c4_random_pseudo_results.txt")
    print(
        "\nTo build the full 2x2 table, join this file's "
        "random_pseudo_coverage_mean against c4_control_results.json's "
        "random_true_coverage (same dataset/method/threshold keys) and "
        "c4_results.json's filtered true/pseudo columns."
    )


if __name__ == "__main__":
    main()