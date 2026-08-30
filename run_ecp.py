"""
C5: Entropy-scaled Conformal Prediction (ECP) for zero-shot CLIP.

Faithful implementation of Kasa et al. (2025), Algorithm 1, adapted to
this project's negatively-oriented LAC nonconformity score.

DERIVATION (documented rather than asserted):

  Kasa's Eq. 4 uses a POSITIVELY-oriented score s(x,y) = pi(x)_y, i.e.
  the raw softmax probability, larger = more likely included:

      C(x) := {y : s(x,y) * max(1, f(u_test)) >= tau_D}

  This project's LAC score is S_LAC(x,y) = 1 - pi(x)_y, negatively
  oriented, with inclusion rule S_LAC <= q_hat. These are the same
  rule with tau_D = 1 - q_hat:

      1 - p_y <= q_hat   <=>   p_y >= 1 - q_hat   <=>   p_y >= tau_D

  Applying Kasa's scaling to the positively-oriented form:

      p_y * max(1, f(u_test)) >= tau_D
      p_y >= tau_D / max(1, f(u_test))

  Equivalently, in this project's negatively-oriented terms:

      S_LAC <= 1 - (1 - q_hat) / max(1, f(u_test))

  NOTE this is a DIVISION of tau_D by the scale factor, NOT a
  multiplication of q_hat by it. An earlier version of this experiment
  multiplied q_hat directly, which is not the same operation and does
  not implement Kasa's method.

SCOPE: LAC only. Kasa's paper states the framework can apply to
negatively-oriented scores such as APS, but does not specify the
mapping. APS/RAPS scores are cumulative sums over sorted
probabilities, with no clean algebraic correspondence to a
per-class positively-oriented softmax score. Rather than invent a
transformation, APS/RAPS are deliberately out of scope here; extending
ECP to them would require deriving (and justifying) a mapping the
source paper does not provide.

Both published scaling variants are evaluated:
    ECP1: f(u) = u        (linear)
    ECP2: f(u) = u^2      (quadratic)

Target labels are used ONLY for final evaluation. The entropy
statistic u_test is computed from unlabelled target covariates alone,
and tau_D comes from source (ImageNet) calibration.

Reads cached embeddings only. No new inference.

Usage:
    python run_ecp.py
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

TARGET_DATASETS = ["imagenet_v2", "imagenet_r", "imagenet_a"]
NUM_DRAWS = 20
UNLABELLED_SIZE = 250   # unlabelled target sample for the entropy statistic
EVAL_SIZE = 500


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row, in nats (Kasa et al. Eq. 5)."""
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


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
        raise ValueError(f"Unsupported target dataset: {dataset}")

    text_emb = model.encode_text(class_names).cpu().numpy()
    probs = _softmax(model.logit_scale * (embeddings @ text_emb.T))
    return probs, labels, len(class_names)


def _lac_sets_with_scaled_threshold(
    eval_probs: np.ndarray, q_hat: float, scale: float
) -> list:
    """
    Build LAC prediction sets under ECP's scaled threshold.

    Implements p_y >= tau_D / max(1, scale), with tau_D = 1 - q_hat.
    Always returns at least the top-1 class, so sets are never empty.
    """
    tau_d = 1.0 - q_hat
    effective_threshold = tau_d / max(1.0, scale)

    included = eval_probs >= effective_threshold
    sets = []
    for i, row in enumerate(included):
        idx = np.where(row)[0]
        sets.append(idx.tolist() if idx.size > 0 else [int(np.argmax(eval_probs[i]))])
    return sets


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

    # Source calibration on ImageNet, full 1000-class space.
    cal_emb, cal_labels = cache.load("clip_imagenet_calibration")
    text_full = model.encode_text(full_class_names).cpu().numpy()
    source_cal_probs = _softmax(model.logit_scale * (cal_emb @ text_full.T))

    lac = create_conformal_method("lac", alpha=alpha)
    lac.calibrate(source_cal_probs, cal_labels)
    q_hat = lac.q_hat
    tau_d = 1.0 - q_hat

    print(f"Source-calibrated LAC: q_hat={q_hat:.6f}, tau_D={tau_d:.6f}\n")

    results = []
    lines = [
        f"{'dataset':>13} {'variant':>8} {'u_test':>8} {'scale':>8} "
        f"{'eff_thresh':>11} {'coverage':>9} {'cov_gap':>9} {'set_size':>9}"
    ]

    for dataset in TARGET_DATASETS:
        probs, labels, n_classes = _load_probs_and_labels(
            dataset, model, full_class_names, cache
        )
        total = probs.shape[0]

        for variant, f_func in [("frozen", None), ("ECP1", lambda u: u), ("ECP2", lambda u: u ** 2)]:
            covs, sizes, u_tests, scales = [], [], [], []

            for draw in range(NUM_DRAWS):
                rng = np.random.default_rng(config.seed.value + draw)
                perm = rng.permutation(total)
                unlabelled_idx = perm[:UNLABELLED_SIZE]
                eval_idx = perm[UNLABELLED_SIZE:UNLABELLED_SIZE + EVAL_SIZE]

                if f_func is None:
                    scale = 1.0
                    u_test = float("nan")
                else:
                    # u_test from UNLABELLED target covariates only.
                    ent = _entropy(probs[unlabelled_idx])
                    u_test = float(np.quantile(ent, 1 - alpha))
                    scale = max(1.0, f_func(u_test))

                eval_probs, eval_labels = probs[eval_idx], labels[eval_idx]
                sets = _lac_sets_with_scaled_threshold(eval_probs, q_hat, scale)
                r = coverage_report(sets, eval_labels, alpha=alpha)

                covs.append(r["empirical_coverage"])
                sizes.append(r["average_set_size"])
                u_tests.append(u_test)
                scales.append(scale)

            mean_cov = float(np.mean(covs))
            mean_scale = float(np.mean(scales))
            row = {
                "dataset": dataset,
                "variant": variant,
                "n_classes": n_classes,
                "mean_u_test": float(np.nanmean(u_tests)),
                "mean_scale": mean_scale,
                "effective_threshold": tau_d / mean_scale,
                "coverage_mean": mean_cov,
                "coverage_std": float(np.std(covs)),
                "coverage_gap": mean_cov - (1 - alpha),
                "set_size_mean": float(np.mean(sizes)),
                "set_size_std": float(np.std(sizes)),
            }
            results.append(row)

            line = (
                f"{dataset:>13} {variant:>8} {row['mean_u_test']:>8.3f} "
                f"{mean_scale:>8.3f} {row['effective_threshold']:>11.6f} "
                f"{mean_cov:>9.4f} {row['coverage_gap']:>+9.4f} "
                f"{row['set_size_mean']:>9.2f}"
            )
            lines.append(line)
            print(line)

    out_dir = Path("results/ecp")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "c5_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"q_hat": q_hat, "tau_D": tau_d, "alpha": alpha, "results": results},
            f, indent=2,
        )
    with open(out_dir / "c5_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWritten to {out_dir}/c5_results.txt")


if __name__ == "__main__":
    main()