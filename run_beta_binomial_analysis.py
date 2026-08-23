"""
Finite-batch (Beta-Binomial) distributional analysis of recalibration-
sweep coverage, replacing the earlier Beta-only comparison in Section
7.4 of the dissertation.

Rationale (per collaborator review): the earlier analysis compared
observed coverage (K_N / 500, a proportion out of a finite 500-example
evaluation batch) directly against Beta(n+1-l, l), the distribution of
the underlying CALIBRATION-CONDITIONAL COVERAGE PROBABILITY p_N. These
are not the same quantity. The correct generative model is:

    p_N ~ Beta(a_N, b_N)                     [calibration-conditional
                                                coverage probability]
    K_N | p_N ~ Binomial(m=500, p_N)         [observed count, out of
                                                the finite evaluation
                                                batch]
    => K_N ~ BetaBinomial(m=500, a_N, b_N)   [marginal distribution of
                                                the observed count]

This script compares each recalibration sweep's 20 realised K_N/500
values against this Beta-Binomial reference, via Kolmogorov-Smirnov
distance, computed and reported DESCRIPTIVELY - not as a formal
hypothesis test - because all 20 draws at a given N share one fixed
500-example evaluation set (Section 4.5.2).

Usage:
    python run_beta_binomial_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import betabinom, kstest

ALPHA = 0.1
EVAL_BATCH_SIZE = 500
N_BUDGETS = [10, 25, 50, 100, 250]

RUNS = [
    ("imagenet_r", "lac", "RECAL-imagenet_r-lac"),
    ("imagenet_r", "aps", "RECAL-imagenet_r-aps-randomized"),
    ("imagenet_r", "raps", "RECAL-imagenet_r-raps-randomized"),
    ("imagenet_a", "lac", "RECAL-imagenet_a-lac"),
    ("imagenet_a", "aps", "RECAL-imagenet_a-aps-randomized"),
    ("imagenet_a", "raps", "RECAL-imagenet_a-raps-randomized"),
]


def main() -> None:
    lines = []
    lines.append(
        f"{'dataset':>12} {'method':>6} {'N':>5} "
        f"{'obs_mean':>9} {'theory_mean':>11} {'KS_dist':>8} {'KS_p':>8}"
    )

    for dataset, method, folder in RUNS:
        path = Path(f"results/{folder}/recovery.json")
        if not path.exists():
            print(f"MISSING: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for n in N_BUDGETS:
            entry = data["results_by_n"][str(n)]
            k_counts = np.round(np.array(entry["coverage_draws"]) * EVAL_BATCH_SIZE).astype(int)

            l = int(np.floor((n + 1) * ALPHA))
            a_param, b_param = n + 1 - l, l

            theory_mean_proportion = betabinom.mean(EVAL_BATCH_SIZE, a_param, b_param) / EVAL_BATCH_SIZE
            observed_mean_proportion = k_counts.mean() / EVAL_BATCH_SIZE

            ks_stat, ks_pvalue = kstest(
                k_counts, betabinom(EVAL_BATCH_SIZE, a_param, b_param).cdf
            )

            line = (
                f"{dataset:>12} {method:>6} {n:>5} "
                f"{observed_mean_proportion:>9.4f} {theory_mean_proportion:>11.4f} "
                f"{ks_stat:>8.4f} {ks_pvalue:>8.4f}"
            )
            lines.append(line)
            print(line)

    output_path = Path("results/distributional_analysis/beta_binomial_verified.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWritten to {output_path}")
    print(
        "\nNOTE: KS distances above are DESCRIPTIVE, per Section 4.5.2's "
        "dependence caveat (20 draws per N share one fixed evaluation "
        "set, so they are not fully independent samples of the "
        "Beta-Binomial reference)."
    )


if __name__ == "__main__":
    main()