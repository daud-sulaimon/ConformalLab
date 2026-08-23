"""
ReCal-CP reporting layer: turns a baseline_evaluation()/recovery_sweep()
result into a printed diagnostic, a saved JSON/CSV, and two plots.

Pure presentation - contains no calibration logic of its own, only
formats what src.conformal.core already computed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def print_diagnostic_report(baseline: dict, sweep: dict) -> None:
    """
    Print the RECAL-CP RELIABILITY REPORT to the console, matching the
    format: baseline diagnostic, recovery table, N* decision.
    """
    gap = baseline["coverage_gap"]
    print("\nRECAL-CP RELIABILITY REPORT")
    print("-" * 32)
    print(f"Method:            {baseline['method'].upper()}")
    print(f"Frozen coverage:   {baseline['empirical_coverage']:.1%}")
    print(f"Nominal coverage:  {baseline['target_coverage']:.1%}")
    print(f"Coverage gap:      {gap:+.1%}")
    print(f"Mean set size:     {baseline['average_set_size']:.2f}")

    print(f"\nRecalibration range: N = {sweep['budgets'][0]} ... {sweep['budgets'][-1]}")
    print("Recovery criterion:  within +/-2pp coverage, SD <= 0.03\n")

    print(f"{'N':>6} {'Mean coverage':>15} {'SD':>8} {'Mean set size':>15}")
    for n in sweep["budgets"]:
        r = sweep["results_by_n"][n]
        print(f"{n:>6} {r['coverage_mean']:>14.1%} {r['coverage_std']:>8.4f} {r['set_size_mean']:>15.2f}")

    status = "NOT REACHED" if sweep["n_star_status"] == "not_reached_within_tested_range" else str(sweep["n_star"])
    print(f"\nN*: {status}")


def save_report(baseline: dict, sweep: dict, output_dir: str | Path) -> None:
    """Save the combined report as JSON and the recovery table as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "sweep": sweep}, f, indent=2)

    with open(output_dir / "recovery_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "coverage_mean", "coverage_std", "set_size_mean", "set_size_std"])
        for n in sweep["budgets"]:
            r = sweep["results_by_n"][n]
            writer.writerow([n, r["coverage_mean"], r["coverage_std"], r["set_size_mean"], r["set_size_std"]])


def plot_recovery_curve(sweep: dict, output_path: str | Path) -> None:
    """Figure 1: coverage vs. calibration budget, with target line and +/-2pp band."""
    ns = sweep["budgets"]
    means = [sweep["results_by_n"][n]["coverage_mean"] for n in ns]
    stds = [sweep["results_by_n"][n]["coverage_std"] for n in ns]
    target = sweep["target_coverage"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ns, means, yerr=stds, marker="o", capsize=3, label="Observed coverage")
    ax.axhline(target, color="red", linestyle="--", linewidth=1, label="Nominal target")
    ax.axhspan(target - 0.02, target + 0.02, color="red", alpha=0.1, label="Recovery band (+/-2pp)")
    ax.set_xlabel("Calibration budget N")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Recovery curve: {sweep['method'].upper()}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_coverage_efficiency(sweep: dict, output_path: str | Path) -> None:
    """Figure 2: coverage vs. mean set size, one point per N, annotated."""
    ns = sweep["budgets"]
    coverages = [sweep["results_by_n"][n]["coverage_mean"] for n in ns]
    sizes = [sweep["results_by_n"][n]["set_size_mean"] for n in ns]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, coverages, marker="o")
    for n, x, y in zip(ns, sizes, coverages):
        ax.annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(sweep["target_coverage"], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean prediction-set size")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Coverage vs. efficiency: {sweep['method'].upper()}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)