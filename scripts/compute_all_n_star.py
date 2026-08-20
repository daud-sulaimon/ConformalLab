"""
Compute N* (per Methodology Section 4.5.1's pre-registered criteria)
for every completed recalibration sweep and print a summary table.

Not a test, not part of the core package - a reporting utility, kept
permanently in scripts/ so the N* table can be regenerated any time
recovery.json files change (e.g. after adding a new seed or dataset).

Usage:
    python scripts/compute_all_n_star.py
"""

import json
from pathlib import Path

from src.metrics.recovery import compute_n_star

TARGET_COVERAGE = 0.90
DATASETS = ["imagenet_r", "imagenet_a"]
METHODS = ["lac", "aps", "raps"]

print(f"{'Dataset':>12} / {'Method':<6}: N*")
print("-" * 32)

for dataset in DATASETS:
    for method in METHODS:
        path = Path(f"results/RECAL-{dataset}-{method}/recovery.json")
        if not path.exists():
            print(f"{dataset:>12} / {method:<6}: MISSING ({path})")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        n_star = compute_n_star(data["results_by_n"], TARGET_COVERAGE)
        label = str(n_star) if n_star is not None else "not reached"
        print(f"{dataset:>12} / {method:<6}: {label}")