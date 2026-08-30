"""
Precision-Selection-Calibration Trade-off figure.

For each conformal method (LAC, APS, RAPS), plots how three quantities
move together as confidence-filtering threshold increases:

  - pseudo-label precision within the filtered subset (left axis)
  - the matched-N selection effect, delta_filter_pseudo (right axis,
    zero line marked - this is the actual result: filtering hurts LAC,
    helps APS/RAPS, and this figure shows both moving together)
  - effective calibration size N (marker size - shrinks as threshold
    rises, visible without a fourth axis)

One panel per method, one line per dataset within each panel, so the
consistency of the effect's DIRECTION within each method (independent
of dataset) is visible at a glance - which is the finding, not any
single number.

Reads results/pseudo_calibration/c4_results.json (precision, N) and
c4_random_pseudo_results.json (delta_filter_pseudo), joined on
(dataset, method, threshold). Excludes threshold=0.0 (no filtering,
not part of the filtering-effect story).

Usage:
    python make_tradeoff_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ["lac", "aps", "raps"]
DATASETS = ["imagenet", "imagenet_v2", "imagenet_r", "imagenet_a"]
DATASET_COLORS = {
    "imagenet": "#1f77b4",
    "imagenet_v2": "#ff7f0e",
    "imagenet_r": "#2ca02c",
    "imagenet_a": "#d62728",
}
DATASET_LABELS = {
    "imagenet": "ImageNet",
    "imagenet_v2": "ImageNet-V2",
    "imagenet_r": "ImageNet-R",
    "imagenet_a": "ImageNet-A",
}


def main() -> None:
    c4_path = Path("results/pseudo_calibration/c4_results.json")
    delta_path = Path("results/pseudo_calibration/c4_random_pseudo_results.json")

    if not c4_path.exists() or not delta_path.exists():
        raise FileNotFoundError(
            "Requires both c4_results.json and c4_random_pseudo_results.json. "
            "Run run_confidence_filtered_pseudo_calibration.py and "
            "run_c4_random_pseudo.py first."
        )

    with open(c4_path, "r", encoding="utf-8") as f:
        c4_rows = json.load(f)
    with open(delta_path, "r", encoding="utf-8") as f:
        delta_rows = json.load(f)

    # Join on (dataset, method, threshold).
    delta_lookup = {
        (r["dataset"], r["method"], r["threshold"]): r for r in delta_rows
    }

    joined = []
    for row in c4_rows:
        if row["threshold"] == 0.0:
            continue
        key = (row["dataset"], row["method"], row["threshold"])
        if key not in delta_lookup:
            continue
        d = delta_lookup[key]
        joined.append({
            "dataset": row["dataset"],
            "method": row["method"],
            "threshold": row["threshold"],
            "precision": row["mean_pseudo_label_precision"],
            "n_eff": row["mean_effective_calibration_n"],
            "delta_filter_pseudo": d["delta_filter_pseudo"],
        })

    if not joined:
        raise RuntimeError("No matching rows found after joining the two result files.")

    # Write the joined dataset for verification / re-use.
    out_dir = Path("results/pseudo_calibration")
    with open(out_dir / "tradeoff_joined_data.json", "w", encoding="utf-8") as f:
        json.dump(joined, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

    for ax, method in zip(axes, METHODS):
        ax_delta = ax.twinx()

        for dataset in DATASETS:
            rows = sorted(
                [r for r in joined if r["method"] == method and r["dataset"] == dataset],
                key=lambda r: r["threshold"],
            )
            if not rows:
                continue
            thresholds = [r["threshold"] for r in rows]
            precisions = [r["precision"] for r in rows]
            deltas = [r["delta_filter_pseudo"] for r in rows]
            sizes = [max(20, r["n_eff"]) for r in rows]  # marker area ~ N

            color = DATASET_COLORS[dataset]

            ax.plot(
                thresholds, precisions, color=color, linestyle="-",
                marker="o", markersize=0, alpha=0.6, linewidth=1.5,
            )
            ax.scatter(
                thresholds, precisions, s=sizes, color=color, alpha=0.6,
                edgecolors="none", zorder=3,
            )

            ax_delta.plot(
                thresholds, deltas, color=color, linestyle="--",
                marker="s", markersize=5, linewidth=1.5,
                label=DATASET_LABELS[dataset],
            )

        ax.set_title(method.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylim(0.2, 1.05)
        ax_delta.axhline(0.0, color="grey", linestyle=":", linewidth=1)
        ax_delta.set_ylim(-0.65, 0.30)

        if method == "lac":
            ax.set_ylabel("Pseudo-label precision\n(solid, circle, size ~ N_eff)")
        if method == "raps":
            ax_delta.set_ylabel("Selection effect, matched N\n(dashed, square)")

    axes[-1].legend(
        loc="lower left", fontsize=8, title="Dataset", framealpha=0.9
    )

    fig.suptitle(
        "Precision rises with confidence threshold for every method (solid) — "
        "but the matched-N calibration effect (dashed) diverges by score",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_dir / "tradeoff_figure.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Figure saved to {out_path}")
    print(f"Joined data saved to {out_dir / 'tradeoff_joined_data.json'}")


if __name__ == "__main__":
    main()