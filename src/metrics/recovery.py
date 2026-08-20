"""
Recovery-criterion utilities for ConformalLab / ReCal-CP.

Implements N* exactly as specified in the dissertation's Methodology
(Section 4.5.1): the smallest tested calibration budget N satisfying
BOTH a coverage-tolerance criterion and a stability criterion,
evaluated in that order. Set size is deliberately excluded from the
definition itself and reported only as a secondary, separate finding
- folding efficiency into the recovery criterion would make any later
efficiency claim circular.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_EPSILON = 0.02
DEFAULT_SIGMA_MAX = 0.03


def compute_n_star(
    recovery_data: dict,
    target_coverage: float,
    epsilon: float = DEFAULT_EPSILON,
    sigma_max: float = DEFAULT_SIGMA_MAX,
) -> Optional[int]:
    """
    Compute N*, the minimum recalibration budget satisfying both the
    coverage-tolerance and stability criteria, per the pre-registered
    definition in Methodology Section 4.5.1.

    N* = min{N : |mean_coverage_N - target| <= epsilon
               AND std_coverage_N <= sigma_max}

    Parameters
    ----------
    recovery_data
        The "results_by_n" dict from a recovery.json file, i.e.
        {str(n): {"coverage_mean": ..., "coverage_std": ..., ...}, ...}.
    target_coverage
        The nominal target, e.g. 0.90 for alpha=0.1.
    epsilon
        Coverage tolerance. Defaults to 0.02 (±2 percentage points),
        matching the dissertation's pre-registered value.
    sigma_max
        Maximum acceptable standard deviation across draws. Defaults
        to 0.03, matching the dissertation's pre-registered value.

    Returns
    -------
    int or None
        The smallest N (from the tested budgets, in ascending order)
        satisfying both criteria simultaneously, or None if no tested
        N satisfies both - in which case N* is reported as "not
        reached within tested range", never extrapolated.

    Examples
    --------
    >>> data = {"10": {"coverage_mean": 0.95, "coverage_std": 0.05},
    ...         "50": {"coverage_mean": 0.91, "coverage_std": 0.02}}
    >>> compute_n_star(data, target_coverage=0.90)
    50
    """
    tested_n_values = sorted(int(n) for n in recovery_data.keys())

    for n in tested_n_values:
        entry = recovery_data[str(n)]
        coverage_ok = abs(entry["coverage_mean"] - target_coverage) <= epsilon
        stability_ok = entry["coverage_std"] <= sigma_max

        if coverage_ok and stability_ok:
            return n

    return None


def summarize_n_star_across_experiments(
    experiment_paths: dict,
    target_coverage: float,
    epsilon: float = DEFAULT_EPSILON,
    sigma_max: float = DEFAULT_SIGMA_MAX,
) -> dict:
    """
    Compute N* for a batch of (dataset, method) recovery.json files.

    Parameters
    ----------
    experiment_paths
        Mapping of a display label (e.g. "imagenet_r/lac") to a loaded
        "results_by_n" dict (as returned by json.load on a
        recovery.json's ["results_by_n"] field).
    target_coverage, epsilon, sigma_max
        As in `compute_n_star`.

    Returns
    -------
    dict
        Mapping of the same labels to their computed N* (int or None).
    """
    return {
        label: compute_n_star(data, target_coverage, epsilon, sigma_max)
        for label, data in experiment_paths.items()
    }