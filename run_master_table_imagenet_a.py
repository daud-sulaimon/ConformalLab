"""
Master ImageNet-A information-regime table.

Single reproducible artefact placing every evaluated target-information
strategy on ONE protocol, with the pre-registered recovery criterion
(Section 4.5.1) applied programmatically rather than by inspection.

PROTOCOL NOTE - this is deliberately distinct from the budget-sweep
protocol used in the RQ3 recalibration curves, and the two must not be
merged. The budget sweep used a fixed 500-example calibration pool and
500-example evaluation set; this comparison uses a 250-example
candidate pool and 500-example evaluation set, because ECP's entropy
statistic and the labelled budgets must draw from the same pool for
the comparison to be fair. Coverage is measurably sensitive to this
choice (LAC at N=250 gives 92.7% under the sweep protocol, 90.3%
here), which is itself a documented finding, not an inconsistency to
be resolved by choosing whichever number is more convenient.

RESOURCE NOTE - the "target labels" column counts labelled target
examples only. ECP consumes unlabelled target covariates (250 for its
entropy statistic), which is a different resource, not a smaller
amount of the same one. The zero-label regimes are therefore described
as satisfying a "label-free recovery condition", NOT as achieving
"N* = 0".

Usage:
    python run_master_table_imagenet_a.py
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

NUM_DRAWS = 20
EVAL_SIZE = 500
CANDIDATE_POOL = 250
LABELLED_BUDGETS = [10, 25, 50, 100, 250]
CONFIDENCE_THRESHOLD = 0.9

# Pre-registered recovery criterion, Section 4.5.1 - fixed before results.
EPSILON = 0.02
SIGMA_MAX = 0.03


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def _lac_sets(eval_probs: np.ndarray, q_hat: float, scale: float = 1.0) -> list:
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

    cal_emb, cal_labels = cache.load("clip_imagenet_calibration")
    text_full = model.encode_text(full_class_names).cpu().numpy()
    source_probs = _softmax(model.logit_scale * (cal_emb @ text_full.T))
    src = create_conformal_method("lac", alpha=alpha)
    src.calibrate(source_probs, cal_labels)
    frozen_q_hat = src.q_hat

    mapping = load_imagenet_a_local_to_full_mapping()
    class_names = [full_class_names[mapping[i]] for i in range(len(mapping))]
    text_active = model.encode_text(class_names).cpu().numpy()
    tgt_emb, tgt_labels = cache.load("clip_imagenet_a_test")
    probs = _softmax(model.logit_scale * (tgt_emb @ text_active.T))
    pseudo_labels = probs.argmax(axis=1)
    top1_conf = probs.max(axis=1)
    total = probs.shape[0]

    acc = float((pseudo_labels == tgt_labels).mean())
    regimes: dict = {}

    for draw in range(NUM_DRAWS):
        rng = np.random.default_rng(config.seed.value + draw)
        perm = rng.permutation(total)
        pool_idx = perm[:CANDIDATE_POOL]
        eval_idx = perm[CANDIDATE_POOL:CANDIDATE_POOL + EVAL_SIZE]
        eval_probs, eval_labels = probs[eval_idx], tgt_labels[eval_idx]

        def record(name, sets, n_labels, order):
            r = coverage_report(sets, eval_labels, alpha=alpha)
            e = regimes.setdefault(
                name, {"cov": [], "size": [], "n_labels": n_labels, "order": order}
            )
            e["cov"].append(r["empirical_coverage"])
            e["size"].append(r["average_set_size"])

        record("Frozen source CP", _lac_sets(eval_probs, frozen_q_hat), 0, 0)

        m = create_conformal_method("lac", alpha=alpha)
        m.calibrate(probs[pool_idx], pseudo_labels[pool_idx])
        record("Pseudo-calibration", _lac_sets(eval_probs, m.q_hat), 0, 1)

        keep = top1_conf[pool_idx] >= CONFIDENCE_THRESHOLD
        if keep.sum() >= 5:
            fidx = pool_idx[keep]
            m = create_conformal_method("lac", alpha=alpha)
            m.calibrate(probs[fidx], pseudo_labels[fidx])
            record(
                f"Filtered pseudo (c>={CONFIDENCE_THRESHOLD})",
                _lac_sets(eval_probs, m.q_hat), 0, 2,
            )

        ent = _entropy(probs[pool_idx])
        u = float(np.quantile(ent, 1 - alpha))
        record("ECP1 (linear)", _lac_sets(eval_probs, frozen_q_hat, max(1.0, u)), 0, 3)
        record("ECP2 (quadratic)", _lac_sets(eval_probs, frozen_q_hat, max(1.0, u ** 2)), 0, 4)

        for n in LABELLED_BUDGETS:
            sub = np.random.default_rng(config.seed.value + draw + n * 7919)
            idx = sub.choice(pool_idx, size=min(n, len(pool_idx)), replace=False)
            m = create_conformal_method("lac", alpha=alpha)
            m.calibrate(probs[idx], tgt_labels[idx])
            record(f"Labelled N={n}", _lac_sets(eval_probs, m.q_hat), n, 5 + n)

    rows = []
    for name, d in regimes.items():
        cov = float(np.mean(d["cov"]))
        sd = float(np.std(d["cov"]))
        gap = abs(cov - target_coverage)
        cov_ok = gap <= EPSILON
        sd_ok = sd <= SIGMA_MAX
        rows.append({
            "regime": name,
            "target_labels": d["n_labels"],
            "coverage": cov,
            "coverage_sd": sd,
            "coverage_gap_pp": gap * 100,
            "set_size": float(np.mean(d["size"])),
            "set_size_sd": float(np.std(d["size"])),
            "meets_coverage_tolerance": cov_ok,
            "meets_stability_tolerance": sd_ok,
            "meets_recovery_criterion": cov_ok and sd_ok,
            "_order": d["order"],
        })
    rows.sort(key=lambda r: r["_order"])

    lines = [
        "ImageNet-A target-information regime comparison",
        f"LAC, alpha={alpha}, {NUM_DRAWS} draws, candidate pool={CANDIDATE_POOL}, "
        f"eval={EVAL_SIZE}, 200-class space",
        f"Zero-shot CLIP top-1 accuracy on this target sample: {acc:.3f}",
        f"Pre-registered recovery criterion: |coverage-{target_coverage:.2f}| <= {EPSILON} "
        f"AND SD <= {SIGMA_MAX}",
        "",
        f"{'Regime':>32} {'Labels':>7} {'Coverage':>9} {'SD':>7} "
        f"{'Gap(pp)':>8} {'SetSize':>8} {'Criterion':>10}",
        "-" * 90,
    ]
    for r in rows:
        lines.append(
            f"{r['regime']:>32} {r['target_labels']:>7} {r['coverage']:>9.4f} "
            f"{r['coverage_sd']:>7.4f} {r['coverage_gap_pp']:>8.2f} "
            f"{r['set_size']:>8.2f} {'PASS' if r['meets_recovery_criterion'] else 'fail':>10}"
        )

    passing_labelled = [
        r["target_labels"] for r in rows
        if r["meets_recovery_criterion"] and r["target_labels"] > 0
    ]
    lines.append("")
    lines.append(
        f"Smallest tested labelled budget meeting the criterion: "
        f"N={min(passing_labelled) if passing_labelled else 'none within tested range'}"
    )
    lines.append(
        "Label-free regimes meeting the criterion: "
        + (", ".join(
            r["regime"] for r in rows
            if r["meets_recovery_criterion"] and r["target_labels"] == 0
        ) or "none")
    )

    out_dir = Path("results/information_regimes")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "master_table_imagenet_a.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "master_table_imagenet_a.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWritten to {out_dir}/master_table_imagenet_a.txt")


if __name__ == "__main__":
    main()