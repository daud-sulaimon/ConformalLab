"""
ReCal-CP core: model-agnostic conformal calibration-resource analysis.

Exposes the reusable engine behind this dissertation's experiments as
a clean public API operating entirely on pre-computed classification
probabilities/logits and integer labels. No CLIP, ImageNet, Hugging
Face, or PyTorch import appears anywhere in this file - this is what
makes the "model-agnostic" claim in the dissertation Abstract and
Section 5 literally true, not just asserted.

Two entry points:
- baseline_evaluation(): frozen-threshold diagnostic (Sections 4.4, 7.2).
- recovery_sweep(): Monte Carlo recalibration budget sweep (Sections 4.4, 7.3).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.conformal.factory import create_conformal_method
from src.metrics.coverage import coverage_report
from src.metrics.recovery import compute_n_star

DEFAULT_BUDGETS = (10, 25, 50, 100, 250)
DEFAULT_DRAWS = 20


def baseline_evaluation(
    source_probs: np.ndarray,
    source_labels: np.ndarray,
    target_probs: np.ndarray,
    target_labels: np.ndarray,
    method: str,
    alpha: float = 0.1,
) -> dict:
    """
    Frozen-threshold diagnostic: calibrate once on source data,
    evaluate unchanged on target data (no recalibration).

    Parameters
    ----------
    source_probs, source_labels
        Calibration data: probabilities (n_source, n_classes), integer
        labels (n_source,).
    target_probs, target_labels
        Evaluation data: probabilities (n_target, n_classes_target),
        integer labels (n_target,). n_classes_target need not equal
        the source's class count - the frozen threshold applies to
        whichever array is passed at prediction time, matching this
        dissertation's own ImageNet-R/A protocol (Section 7.2).
    method
        "lac", "aps", or "raps".
    alpha
        Miscoverage rate. Defaults to 0.1.

    Returns
    -------
    dict
        {"method", "target_coverage", "empirical_coverage",
         "average_set_size", "num_samples", "coverage_gap"}.

    Examples
    --------
    >>> report = baseline_evaluation(src_p, src_y, tgt_p, tgt_y, method="aps")
    >>> report["coverage_gap"]
    -0.194
    """
    conformal_method = create_conformal_method(method, alpha=alpha)
    conformal_method.calibrate(source_probs, source_labels)
    prediction_sets = conformal_method.predict_sets(target_probs)

    report = coverage_report(prediction_sets, target_labels, alpha=alpha)
    report["method"] = method
    report["coverage_gap"] = report["empirical_coverage"] - report["target_coverage"]
    return report


def recovery_sweep(
    target_logits: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
    *,
    target_probs: Optional[np.ndarray] = None,
    method: str = "lac",
    alpha: float = 0.1,
    budgets=DEFAULT_BUDGETS,
    draws: int = DEFAULT_DRAWS,
    recal_pool_size: Optional[int] = None,
    eval_set_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo target-domain recalibration budget sweep (Sections
    4.4, 7.3) as a model-agnostic function on raw arrays.

    Accepts EITHER target_probs (already softmax-normalised) OR
    target_logits (raw, normalised internally) - use whichever you have.
    Operates entirely on the target-domain pool, split internally into
    a fixed calibration pool and a fixed evaluation set.

    Parameters
    ----------
    target_logits, target_labels
        Raw logits and integer labels for the target-domain pool.
    target_probs
        Pre-normalised probabilities, alternative to target_logits.
    method
        "lac", "aps", or "raps".
    alpha
        Miscoverage rate. Defaults to 0.1.
    budgets
        Calibration sample sizes to sweep. Defaults to (10,25,50,100,250).
    draws
        Repeated draws per budget. Defaults to 20.
    recal_pool_size, eval_set_size
        Sizes of the two fixed splits carved from the target pool.
        Default to half the available data each.
    seed
        Random seed controlling the pool/eval split and all draws.

    Returns
    -------
    dict
        {"method", "alpha", "target_coverage", "budgets", "draws",
         "results_by_n": {N: {...}}, "n_star", "n_star_status"}.

    Examples
    --------
    >>> result = recovery_sweep(target_probs=probs, target_labels=labels,
    ...                          method="aps", budgets=[10, 50, 250])
    >>> result["n_star_status"]
    'not_reached_within_tested_range'
    """
    if target_probs is None:
        if target_logits is None or target_labels is None:
            raise ValueError(
                "Must provide either target_probs or (target_logits, target_labels)."
            )
        exp_logits = np.exp(target_logits - target_logits.max(axis=1, keepdims=True))
        target_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    if target_labels is None:
        raise ValueError("target_labels is required.")

    total_available = target_probs.shape[0]
    pool_size = recal_pool_size or total_available // 2
    eval_size = eval_set_size or total_available // 2

    if pool_size + eval_size > total_available:
        raise ValueError(
            f"pool_size ({pool_size}) + eval_size ({eval_size}) exceeds "
            f"available target examples ({total_available})."
        )

    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(total_available)
    pool_indices = all_indices[:pool_size]
    eval_indices = all_indices[pool_size:pool_size + eval_size]

    pool_probs, pool_labels = target_probs[pool_indices], target_labels[pool_indices]
    eval_probs, eval_labels = target_probs[eval_indices], target_labels[eval_indices]

    results_by_n = {}
    for n in budgets:
        if n > pool_size:
            raise ValueError(f"Budget N={n} exceeds recalibration pool size ({pool_size}).")

        draw_coverages, draw_set_sizes = [], []
        for draw in range(draws):
            draw_rng = np.random.default_rng(seed + n * 1000 + draw)
            sample_indices = draw_rng.choice(pool_size, size=n, replace=False)

            method_obj = create_conformal_method(method, alpha=alpha)
            method_obj.calibrate(pool_probs[sample_indices], pool_labels[sample_indices])
            prediction_sets = method_obj.predict_sets(eval_probs)

            report = coverage_report(prediction_sets, eval_labels, alpha=alpha)
            draw_coverages.append(report["empirical_coverage"])
            draw_set_sizes.append(report["average_set_size"])

        results_by_n[n] = {
            "coverage_mean": float(np.mean(draw_coverages)),
            "coverage_std": float(np.std(draw_coverages)),
            "set_size_mean": float(np.mean(draw_set_sizes)),
            "set_size_std": float(np.std(draw_set_sizes)),
            "coverage_draws": [float(c) for c in draw_coverages],
            "set_size_draws": [float(s) for s in draw_set_sizes],
        }

    n_star = compute_n_star({str(n): v for n, v in results_by_n.items()}, target_coverage=1 - alpha)

    return {
        "method": method,
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "budgets": list(budgets),
        "draws": draws,
        "recal_pool_size": pool_size,
        "eval_set_size": eval_size,
        "results_by_n": results_by_n,
        "n_star": n_star,
        "n_star_status": "reached" if n_star is not None else "not_reached_within_tested_range",
    }