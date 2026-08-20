"""
Distributional analysis of recalibration-sweep coverage against the
theoretical Beta reference for split-CP calibration-conditional
coverage.

For a conformal method calibrated on n exchangeable examples at
miscoverage alpha, coverage conditional on that specific calibration
set follows Beta(n+1-l, l), where l = floor((n+1)*alpha) (Vovk, 2012;
also see Angelopoulos & Bates, 2023, Section 3). This script compares
each recalibration sweep's 20 realised draw-level coverages against
that reference.

IMPORTANT CAVEAT (see also dissertation Methodology): the theoretical
Beta reference describes the distribution of coverage over independent
draws of the calibration set, evaluated against an independent draw of
the SAME SIZE from the target population. In this project's sweep, all
20 draws at a given N share ONE FIXED 500-example evaluation set. The
20 coverage observations are therefore repeated measurements against a
common target sample, not fully independent samples of the
unconditional coverage distribution. This is documented explicitly so
the KS comparison below is read as a DESCRIPTIVE distributional check,
not a formal independence-assuming hypothesis test, and a KS result
(of any kind) is NOT interpreted as proving or disproving conformal
validity - it only reports whether the observed draws are compatible
with the reference distribution's shape under this test's assumptions.

Run once per (dataset, method); produces console output + one plot per
combination. Does not modify or re-run any experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, kstest

ALPHA = 0.1
DATASETS = ["imagenet_r", "imagenet_a"]
METHODS = ["lac", "aps", "raps"]
N_BUDGETS = [10, 25, 50, 100, 250]

OUTPUT_DIR = Path("results/distributional_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLAG_THRESHOLD = 0.35  # KS distance above this is flagged for manual review, not declared a "failure"


def analyze_one(dataset: str, method: str) -> None:
    path = Path(f"results/RECAL-{dataset}-{method}/recovery.json")
    if not path.exists():
        print(f"MISSING: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"{dataset} / {method}")
    print(f"{'=' * 70}")

    fig, axes = plt.subplots(1, len(N_BUDGETS), figsize=(4 * len(N_BUDGETS), 4), sharey=True)

    for ax, n in zip(axes, N_BUDGETS):
        entry = data["results_by_n"][str(n)]
        draws = np.array(entry["coverage_draws"])

        l = int(np.floor((n + 1) * ALPHA))
        a_param, b_param = n + 1 - l, l

        theoretical_mean = beta.mean(a_param, b_param)
        theoretical_std = beta.std(a_param, b_param)
        observed_mean = draws.mean()
        observed_std = draws.std()

        # Descriptive KS distance against the theoretical CDF.
        # p-value is reported but NOT used as a pass/fail gate, per the
        # dependence caveat above.
        ks_stat, ks_pvalue = kstest(draws, beta(a_param, b_param).cdf)

        flag = " <-- FLAGGED FOR REVIEW" if ks_stat > FLAG_THRESHOLD else ""

        print(
            f"  N={n:>4}: observed mean={observed_mean:.4f} (theory={theoretical_mean:.4f}), "
            f"observed std={observed_std:.4f} (theory={theoretical_std:.4f}), "
            f"KS distance={ks_stat:.4f} (descriptive p={ks_pvalue:.4f}){flag}"
        )

        # Plot: empirical draws (rug + histogram) vs theoretical Beta pdf.
        x = np.linspace(max(0, theoretical_mean - 5 * theoretical_std), 1, 300)
        ax.plot(x, beta.pdf(x, a_param, b_param), label="Theoretical Beta", color="black")
        ax.hist(draws, bins=8, density=True, alpha=0.5, label="Observed draws", color="steelblue")
        ax.axvline(1 - ALPHA, color="red", linestyle="--", linewidth=1, label="Nominal target")
        ax.set_title(f"N={n}\nKS={ks_stat:.3f}")
        ax.set_xlabel("Coverage")

    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{dataset} / {method}: observed draws vs. theoretical Beta reference")
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"{dataset}_{method}_beta_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main() -> None:
    for dataset in DATASETS:
        for method in METHODS:
            analyze_one(dataset, method)

    print(f"\n{'=' * 70}")
    print(
        "NOTE: KS distances above are DESCRIPTIVE. All 20 draws per N share a "
        "single fixed evaluation set, so this is not a formal independence-"
        "assuming hypothesis test. Flagged cases warrant investigation, not "
        "an automatic conclusion that conformal validity was violated."
    )


if __name__ == "__main__":
    main()