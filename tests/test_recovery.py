"""Tests for src.metrics.recovery."""

from src.metrics.recovery import compute_n_star, summarize_n_star_across_experiments


def test_n_star_finds_smallest_satisfying_n():
    data = {
        "10": {"coverage_mean": 0.95, "coverage_std": 0.06},
        "25": {"coverage_mean": 0.93, "coverage_std": 0.04},
        "50": {"coverage_mean": 0.91, "coverage_std": 0.02},
        "100": {"coverage_mean": 0.905, "coverage_std": 0.015},
    }
    assert compute_n_star(data, target_coverage=0.90) == 50


def test_n_star_requires_both_criteria_simultaneously():
    # N=25 satisfies coverage tolerance but not stability - must be skipped.
    data = {
        "25": {"coverage_mean": 0.905, "coverage_std": 0.10},
        "100": {"coverage_mean": 0.91, "coverage_std": 0.02},
    }
    assert compute_n_star(data, target_coverage=0.90) == 100


def test_n_star_returns_none_when_never_satisfied():
    data = {
        "10": {"coverage_mean": 0.70, "coverage_std": 0.05},
        "250": {"coverage_mean": 0.71, "coverage_std": 0.02},
    }
    assert compute_n_star(data, target_coverage=0.90) is None


def test_n_star_respects_custom_thresholds():
    data = {"50": {"coverage_mean": 0.85, "coverage_std": 0.02}}
    # Default epsilon=0.02 rejects this (0.05 gap); a looser epsilon accepts it.
    assert compute_n_star(data, target_coverage=0.90) is None
    assert compute_n_star(data, target_coverage=0.90, epsilon=0.06) == 50


def test_summarize_across_experiments():
    experiments = {
        "imagenet_r/lac": {"250": {"coverage_mean": 0.899, "coverage_std": 0.019}},
        "imagenet_a/lac": {"250": {"coverage_mean": 0.711, "coverage_std": 0.033}},
    }
    result = summarize_n_star_across_experiments(experiments, target_coverage=0.90)
    assert result["imagenet_r/lac"] == 250
    assert result["imagenet_a/lac"] is None