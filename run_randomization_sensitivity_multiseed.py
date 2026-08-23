"""
Multi-seed verification of the deterministic-vs-randomised APS/RAPS
sensitivity check. Runs the randomised variant across 5 independent
seeds (the deterministic variant has no randomness, so it only needs
one run) and reports mean +/- std, matching the same repeated-draw
discipline used throughout the recalibration sweep (Section 4.4).

Reads only cached embeddings - no new CLIP inference or streaming.

Usage:
    python run_randomization_sensitivity_multiseed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.conformal.aps import APSConformalMethod
from src.conformal.raps import RAPSConformalMethod
from src.datasets.imagenet_class_names import load_imagenet_class_names
from src.embeddings.cache import EmbeddingCache
from src.metrics.coverage import coverage_report
from src.models.manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

import src.datasets.imagenet  # noqa: F401
import src.datasets.imagenet_a  # noqa: F401
import src.datasets.imagenet_r  # noqa: F401
import src.models.clip_model  # noqa: F401

logger = get_logger(__name__)

SEEDS = [0, 1, 2, 3, 4]


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _get_class_mapping_names(dataset: str, full_class_names: list) -> list:
    if dataset == "imagenet_r":
        from src.datasets.imagenet_r_class_mapping import load_imagenet_r_local_to_full_mapping
        mapping = load_imagenet_r_local_to_full_mapping()
    elif dataset == "imagenet_a":
        from src.datasets.imagenet_a_class_mapping import load_imagenet_a_local_to_full_mapping
        mapping = load_imagenet_a_local_to_full_mapping()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return [full_class_names[mapping[i]] for i in range(len(mapping))]


def main() -> None:
    configure_logging()
    config = load_config("configs/default.yaml")
    set_seed(config.seed.value)

    model = ModelManager(config.model.name)
    model.load()

    full_class_names = load_imagenet_class_names()
    cache = EmbeddingCache()

    calibration_embeddings, calibration_labels = cache.load(
        f"{config.model.name}_imagenet_calibration"
    )
    text_embeddings_full = model.encode_text(full_class_names).cpu().numpy()
    calibration_probs_full = _softmax(
        model.logit_scale * (calibration_embeddings @ text_embeddings_full.T)
    )

    # Pre-load shift data once.
    shift_data = {}
    for dataset in ["imagenet_r", "imagenet_a"]:
        active_class_names = _get_class_mapping_names(dataset, full_class_names)
        text_embeddings_active = model.encode_text(active_class_names).cpu().numpy()
        shift_embeddings, shift_labels = cache.load(f"{config.model.name}_{dataset}_test")
        shift_probs = _softmax(model.logit_scale * (shift_embeddings @ text_embeddings_active.T))
        shift_data[dataset] = (shift_probs, shift_labels)

    results = []

    for method_name, method_class in [("aps", APSConformalMethod), ("raps", RAPSConformalMethod)]:
        # Deterministic: single run, no seed variation possible.
        det_method = method_class(alpha=config.calibration.alpha, randomize=False)
        det_method.calibrate(calibration_probs_full, calibration_labels)
        det_q_hat = det_method.q_hat

        for dataset in ["imagenet_r", "imagenet_a"]:
            shift_probs, shift_labels = shift_data[dataset]
            eval_method = method_class(alpha=config.calibration.alpha, randomize=False)
            eval_method._q_hat = det_q_hat
            report = coverage_report(
                eval_method.predict_sets(shift_probs), shift_labels, alpha=config.calibration.alpha
            )
            results.append({
                "dataset": dataset, "method": method_name, "randomize": False,
                "seed": None, "coverage": report["empirical_coverage"],
                "avg_set_size": report["average_set_size"],
            })

        # Randomised: 5 independent seeds, report mean +/- std per dataset.
        for dataset in ["imagenet_r", "imagenet_a"]:
            shift_probs, shift_labels = shift_data[dataset]
            seed_coverages, seed_set_sizes = [], []

            for seed in SEEDS:
                rand_method = method_class(alpha=config.calibration.alpha, randomize=True, seed=seed)
                rand_method.calibrate(calibration_probs_full, calibration_labels)

                eval_method = method_class(alpha=config.calibration.alpha, randomize=True, seed=seed)
                eval_method._q_hat = rand_method.q_hat
                report = coverage_report(
                    eval_method.predict_sets(shift_probs), shift_labels, alpha=config.calibration.alpha
                )
                seed_coverages.append(report["empirical_coverage"])
                seed_set_sizes.append(report["average_set_size"])
                results.append({
                    "dataset": dataset, "method": method_name, "randomize": True,
                    "seed": seed, "coverage": report["empirical_coverage"],
                    "avg_set_size": report["average_set_size"],
                })

            logger.info(
                f"{dataset}/{method_name} randomize=True (5 seeds): "
                f"coverage={np.mean(seed_coverages):.4f}+/-{np.std(seed_coverages):.4f} "
                f"set_size={np.mean(seed_set_sizes):.2f}+/-{np.std(seed_set_sizes):.2f}"
            )

    output_dir = Path("results/randomization_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "sensitivity_multiseed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [f"{'dataset':>12} {'method':>6} {'randomize':>10} {'seed':>5} {'coverage':>10} {'set_size':>10}"]
    for r in results:
        lines.append(
            f"{r['dataset']:>12} {r['method']:>6} {str(r['randomize']):>10} "
            f"{str(r['seed']):>5} {r['coverage']:>10.4f} {r['avg_set_size']:>10.2f}"
        )
    lines.append("\n--- Randomised summary (mean +/- std across 5 seeds) ---")
    for method_name in ["aps", "raps"]:
        for dataset in ["imagenet_r", "imagenet_a"]:
            rows = [r for r in results if r["method"] == method_name and r["dataset"] == dataset and r["randomize"]]
            covs = [r["coverage"] for r in rows]
            sizes = [r["avg_set_size"] for r in rows]
            lines.append(
                f"{dataset:>12} {method_name:>6}: coverage={np.mean(covs):.4f}+/-{np.std(covs):.4f} "
                f"set_size={np.mean(sizes):.2f}+/-{np.std(sizes):.2f}"
            )

    with open(output_dir / "sensitivity_multiseed.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWritten to {output_dir}/sensitivity_multiseed.txt")


if __name__ == "__main__":
    main()