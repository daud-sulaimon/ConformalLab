"""Tests for run_recalibration_sweep.py's core sweep logic, using
synthetic data (no cached embeddings or network required)."""

import numpy as np

from run_recalibration_sweep import _run_sweep_for_dataset


def _make_synthetic_probs_and_labels(n, num_classes=20, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n)
    probs = rng.uniform(0.001, 0.05, size=(n, num_classes))
    probs[np.arange(n), labels] += rng.uniform(0.2, 0.6, size=n)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs, labels


def test_sweep_produces_results_for_every_n_budget():
    probs, labels = _make_synthetic_probs_and_labels(n=1000)
    results = _run_sweep_for_dataset(probs, labels, method_name="lac", alpha=0.1, seed=42)

    assert set(results.keys()) == {10, 25, 50, 100, 250}
    for n, r in results.items():
        assert "coverage_mean" in r
        assert "coverage_std" in r
        assert "set_size_mean" in r
        assert "set_size_std" in r
        assert r["num_draws"] == 20


def test_sweep_raises_with_insufficient_data():
    probs, labels = _make_synthetic_probs_and_labels(n=100)  # too few
    try:
        _run_sweep_for_dataset(probs, labels, method_name="lac", alpha=0.1, seed=42)
        assert False, "Expected ValueError for insufficient data"
    except ValueError as e:
        assert "at least" in str(e)


def test_sweep_is_reproducible_with_same_seed():
    probs, labels = _make_synthetic_probs_and_labels(n=1000)
    results_a = _run_sweep_for_dataset(probs, labels, method_name="aps", alpha=0.1, seed=7)
    results_b = _run_sweep_for_dataset(probs, labels, method_name="aps", alpha=0.1, seed=7)

    for n in results_a:
        assert results_a[n]["coverage_mean"] == results_b[n]["coverage_mean"]


def test_larger_n_generally_reduces_variance():
    """With more calibration data, the threshold should be more stable
    across repeated draws, so std should generally trend downward."""
    probs, labels = _make_synthetic_probs_and_labels(n=1000, seed=1)
    results = _run_sweep_for_dataset(probs, labels, method_name="lac", alpha=0.1, seed=42)

    std_at_10 = results[10]["coverage_std"]
    std_at_250 = results[250]["coverage_std"]
    assert std_at_250 <= std_at_10