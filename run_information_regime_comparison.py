"""
Information-regime comparison on ImageNet-A.

Places all evaluated calibration strategies on ONE common evaluation
protocol so coverage and set size are directly comparable: same
dataset, same 200-class space, same alpha, same LAC method, same
20-draw structure, same 500-example evaluation splits.

The five regimes, ordered by target-label cost:

  1. Frozen source CP           0 target labels
  2. ECP1 / ECP2                0 target labels (unlabelled entropy)
  3. Pseudo-calibration         0 target labels (model-generated labels)
  4. Confidence-filtered pseudo 0 target labels (selected pseudo-labels)
  5. Target recalibration       N in {10,25,50,100,250} true labels

Everything is recomputed here from cached embeddings rather than
joined from prior result files, so no cross-experiment protocol
mismatch can creep in. Prior scripts used differing calibration
pool sizes and split structures; this script fixes one protocol and
applies it to every regime.

LAC only - ECP's mapping is derived for LAC alone (see run_ecp.py).

Usage:
    python run_information_regime_comparison.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.conformal.factory import create_conformal_method
from src.datasets.imagenet_a_class_mapping import (
    load_imagenet_a_local_to_full_mapping,
)
from src.datasets.imagenet_class_names import load_imagenet_class_names
from src.embeddings.cache import EmbeddingCache
from src.metrics.coverage import coverage_report
from src.models.manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

import src.models.clip_model  # noqa: F401

logger = get_logger(__name__)

DATASET = "imagenet_a"
NUM_DRAWS = 20
EVAL_SIZE = 500
CALIBRATION_POOL = 250          # target-domain pool for label-using regimes
UNLABELLED_SIZE = 250           # unlabelled sample for ECP's entropy statistic
LABELLED_BUDGETS = [10, 25, 50, 100, 250]
CONFIDENCE_THRESHOLD = 0.9      # the strictest tested in C4


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def _lac_sets(eval_probs: np.ndarray, q_hat: float, scale: float = 1.0) -> list:
    """LAC prediction sets, optionally with ECP's scaled threshold."""
    tau_d = 1.0 - q_hat
    threshold = tau_d / max(1.0, scale)
    included = eval_probs >= threshold
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
    target_coverage = 1 - alpha

    model = ModelManager("clip")
    model.load()
    full_class_names = load_imagenet_class_names()
    cache = EmbeddingCache()

    # Source calibration (ImageNet, 1000-class space) -> frozen q_hat.
    cal_emb, cal_labels = cache.load("clip_imagenet_calibration")
    text_full = model.encode_text(full_class_names).cpu().numpy()
    source_probs = _softmax(model.logit_scale * (cal_emb @ text_full.T))
    source_lac = create_conformal_method("lac", alpha=alpha)
    source_lac.calibrate(source_probs, cal_labels)
    frozen_q_hat = source_lac.q_hat

    # Target data (ImageNet-A, 200-class restricted space).
    mapping = load_imagenet_a_local_to_full_mapping()
    class_names = [full_class_names[mapping[i]] for i in range(len(mapping))]
    text_active = model.encode_text(class_names).cpu().numpy()
    tgt_emb, tgt_labels = cache.load(f"clip_{DATASET}_test")
    probs = _softmax(model.logit_scale * (tgt_emb @ text_active.T))
    pseudo_labels = probs.argmax(axis=1)
    top1_conf = probs.max(axis=1)
    total = probs.shape[0]

    regimes: dict = {}

    for draw in range(NUM_DRAWS):
        rng = np.random.default_rng(config.seed.value + draw)
        perm = rng.permutation(total)
        pool_idx = perm[:CALIBRATION_POOL]
        eval_idx = perm[CALIBRATION_POOL:CALIBRATION_POOL + EVAL_SIZE]
        eval_probs, eval_labels = probs[eval_idx], tgt_labels[eval_idx]

        def record(name: str, sets, n_labels: int):
            r = coverage_report(sets, eval_labels, alpha=alpha)
            regimes.setdefault(name, {"cov": [], "size": [], "n_labels": n_labels})
            regimes[name]["cov"].append(r["empirical_coverage"])
            regimes[name]["size"].append(r["average_set_size"])

        # 1. Frozen source CP
        record("frozen_source", _lac_sets(eval_probs, frozen_q_hat), 0)

        # 2. ECP1 / ECP2 - entropy from unlabelled pool
        ent = _entropy(probs[pool_idx])
        u_test = float(np.quantile(ent, 1 - alpha))
        record("ECP1", _lac_sets(eval_probs, frozen_q_hat, max(1.0, u_test)), 0)
        record("ECP2", _lac_sets(eval_probs, frozen_q_hat, max(1.0, u_test ** 2)), 0)

        # 3. Pseudo-calibration on the full pool
        m = create_conformal_method("lac", alpha=alpha)
        m.calibrate(probs[pool_idx], pseudo_labels[pool_idx])
        record("pseudo_calibration", _lac_sets(eval_probs, m.q_hat), 0)

        # 4. Confidence-filtered pseudo-calibration
        keep = top1_conf[pool_idx] >= CONFIDENCE_THRESHOLD
        if keep.sum() >= 5:
            fidx = pool_idx[keep]
            m = create_conformal_method("lac", alpha=alpha)
            m.calibrate(probs[fidx], pseudo_labels[fidx])
            record(f"pseudo_filtered_{CONFIDENCE_THRESHOLD}", _lac_sets(eval_probs, m.q_hat), 0)

        # 5. Target recalibration at each labelled budget
        for n in LABELLED_BUDGETS:
            sub_rng = np.random.default_rng(config.seed.value + draw + n * 7919)
            sub_idx = sub_rng.choice(pool_idx, size=min(n, len(pool_idx)), replace=False)
            m = create_conformal_method("lac", alpha=alpha)
            m.calibrate(probs[sub_idx], tgt_labels[sub_idx])
            record(f"labelled_N{n}", _lac_sets(eval_probs, m.q_hat), n)

    # Report, ordered by coverage.
    rows = []
    for name, d in regimes.items():
        rows.append({
            "regime": name,
            "target_labels_used": d["n_labels"],
            "coverage_mean": float(np.mean(d["cov"])),
            "coverage_std": float(np.std(d["cov"])),
            "set_size_mean": float(np.mean(d["size"])),
            "set_size_std": float(np.std(d["size"])),
            "coverage_gap": float(np.mean(d["cov"])) - target_coverage,
        })
    rows.sort(key=lambda r: -r["coverage_mean"])

    lines = [
        f"ImageNet-A, LAC, alpha={alpha}, {NUM_DRAWS} draws, "
        f"eval N={EVAL_SIZE}, 200-class space",
        "",
        f"{'regime':>26} {'labels':>7} {'coverage':>10} {'cov_sd':>8} "
        f"{'set_size':>9} {'size_sd':>8} {'gap':>8}",
    ]
    for r in rows:
        lines.append(
            f"{r['regime']:>26} {r['target_labels_used']:>7} "
            f"{r['coverage_mean']:>10.4f} {r['coverage_std']:>8.4f} "
            f"{r['set_size_mean']:>9.2f} {r['set_size_std']:>8.2f} "
            f"{r['coverage_gap']:>+8.4f}"
        )

    out_dir = Path("results/information_regimes")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "imagenet_a_comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "imagenet_a_comparison.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWritten to {out_dir}/imagenet_a_comparison.txt")


if __name__ == "__main__":
    main()