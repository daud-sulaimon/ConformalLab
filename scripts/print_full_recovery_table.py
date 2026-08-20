"""One-off: print the full recovery table (mean + std at every N) for
all six dataset/method recalibration sweeps, for manual verification."""

import json

for dataset in ["imagenet_r", "imagenet_a"]:
    for method in ["lac", "aps", "raps"]:
        with open(f"results/RECAL-{dataset}-{method}/recovery.json") as f:
            data = json.load(f)
        print(f"\n{dataset} / {method}")
        for n in ["10", "25", "50", "100", "250"]:
            e = data["results_by_n"][n]
            print(f"  N={n:>4}: mean={e['coverage_mean']:.4f}  std={e['coverage_std']:.4f}")