"""
Deterministic-vs-randomised APS/RAPS sensitivity check.

This project's primary results (Sections 7.1-7.4 of the dissertation)
use the deterministic/inclusive APS and RAPS score variants, calibrated
ONCE on ImageNet's full 1000-class space, with that frozen threshold
then applied directly (not recalibrated) to R/A's 200-class-restricted
probability space - matching run_shift_eval.py's established protocol
exactly, via the same private-attribute q_hat injection technique.

This script reproduces that identical protocol for both the
deterministic (primary) and randomised (sensitivity-check) score
variants, so the comparison isolates the effect of randomisation alone
and does not also change the calibration protocol.

Does NOT replace the primary results. Reads only cached embeddings -
no new CLIP inference or streaming.

Usage:
    python run_randomization_sensitivity.py
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

    # Calibration: ONCE, on ImageNet's full 1000-class space - matching
    # run_split_cp.py exactly. calibration_labels are valid indices into
    # this full space, so no restriction/mismatch here.
    calibration_embeddings, calibration_labels = cache.load(
        f"{config.model.name}_imagenet_calibration"
    )
    text_embeddings_full = model.encode_text(full_class_names).cpu().numpy()
    calibration_probs_full = _softmax(
        model.logit_scale * (calibration_embeddings @ text_embeddings_full.T)
    )

    results = []

    for method_name, method_class in [("aps", APSConformalMethod), ("raps", RAPSConformalMethod)]:
        for randomize in [False, True]:
            method = method_class(alpha=config.calibration.alpha, randomize=randomize, seed=0)
            method.calibrate(calibration_probs_full, calibration_labels)
            q_hat = method.q_hat

            for dataset in ["imagenet_r", "imagenet_a"]:
                active_class_names = _get_class_mapping_names(dataset, full_class_names)
                text_embeddings_active = model.encode_text(active_class_names).cpu().numpy()

                shift_embeddings, shift_labels = cache.load(f"{config.model.name}_{dataset}_test")
                shift_probs = _softmax(
                    model.logit_scale * (shift_embeddings @ text_embeddings_active.T)
                )

                # Match run_shift_eval.py's established protocol exactly:
                # inject the frozen, full-class-calibrated q_hat directly
                # rather than recalibrating against the restricted space.
                eval_method = method_class(
                    alpha=config.calibration.alpha, randomize=randomize, seed=0
                )
                eval_method._q_hat = q_hat  # deliberate injection, matching run_shift_eval.py
                prediction_sets = eval_method.predict_sets(shift_probs)
                report = coverage_report(
                    prediction_sets, shift_labels, alpha=config.calibration.alpha
                )

                row = {
                    "dataset": dataset,
                    "method": method_name,
                    "randomize": randomize,
                    "coverage": report["empirical_coverage"],
                    "avg_set_size": report["average_set_size"],
                    "num_classes": len(active_class_names),
                }
                results.append(row)
                logger.info(
                    f"{dataset}/{method_name} randomize={randomize}: "
                    f"coverage={row['coverage']:.4f} set_size={row['avg_set_size']:.2f}"
                )

    output_dir = Path("results/randomization_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sensitivity_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [f"{'dataset':>12} {'method':>6} {'randomize':>10} {'coverage':>10} {'set_size':>10}"]
    for r in results:
        lines.append(
            f"{r['dataset']:>12} {r['method']:>6} {str(r['randomize']):>10} "
            f"{r['coverage']:>10.4f} {r['avg_set_size']:>10.2f}"
        )
    with open(output_dir / "sensitivity_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWritten to {output_path} and sensitivity_results.txt")


if __name__ == "__main__":
    main()