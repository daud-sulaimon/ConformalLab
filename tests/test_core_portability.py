"""
Model-agnostic portability test for ReCal-CP's core engine.

Proves the dissertation's claim that the core analysis operates on
arbitrary classification probability arrays, independent of CLIP or
ImageNet. No CLIP, ImageNet, or Hugging Face import appears anywhere
in this file.
"""

import numpy as np

from src.conformal.core import baseline_evaluation, recovery_sweep


def _synthetic_probs_and_labels(n, num_classes, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n)
    probs = rng.uniform(0.001, 0.05, size=(n, num_classes))
    probs[np.arange(n), labels] += rng.uniform(0.2, 0.6, size=n)
    return probs / probs.sum(axis=1, keepdims=True), labels


def test_baseline_evaluation_runs_on_synthetic_non_clip_data():
    source_probs, source_labels = _synthetic_probs_and_labels(500, num_classes=15, seed=0)
    target_probs, target_labels = _synthetic_probs_and_labels(500, num_classes=15, seed=1)

    report = baseline_evaluation(
        source_probs, source_labels, target_probs, target_labels, method="lac", alpha=0.1
    )
    assert 0.0 <= report["empirical_coverage"] <= 1.0
    assert report["average_set_size"] > 0


def test_recovery_sweep_runs_on_synthetic_non_clip_data():
    target_probs, target_labels = _synthetic_probs_and_labels(1000, num_classes=15, seed=2)

    result = recovery_sweep(
        target_probs=target_probs, target_labels=target_labels,
        method="aps", alpha=0.1, budgets=[10, 50], draws=5,
    )
    assert set(result["results_by_n"].keys()) == {10, 50}
    assert result["n_star_status"] in {"reached", "not_reached_within_tested_range"}


def test_recovery_sweep_accepts_raw_logits_not_only_probabilities():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(500, 10))
    labels = rng.integers(0, 10, size=500)

    result = recovery_sweep(
        target_logits=logits, target_labels=labels, method="lac", alpha=0.1, budgets=[10], draws=5,
    )
    assert 10 in result["results_by_n"]