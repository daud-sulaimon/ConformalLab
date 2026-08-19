"""
Monte Carlo target-domain recalibration budget sweep for ConformalLab.

The core RQ3 deliverable: for a chosen shift dataset (ImageNet-R or
ImageNet-A) and a chosen conformal method (LAC/APS/RAPS), measures how
much labelled target-domain data is needed to recalibrate and recover
coverage close to the nominal target.

Design: the dataset's cached embeddings are split ONCE into a fixed
500-example "recalibration pool" (from which calibration samples of
size N are drawn) and a fixed, constant 500-example "evaluation set"
(never touched for calibration, always used to measure recovered
coverage). This keeps the evaluation set's composition identical
across every N and every repeated draw, so recovery curves reflect
only the calibration budget - not a shrinking or shifting test pool.

For each N in {10, 25, 50, 100, 250}, 20 independent random N-sized
samples are drawn from the recalibration pool (without replacement
within each draw; draws are independent of each other), the method is
freshly calibrated on each sample (never reusing the frozen ImageNet
threshold), and coverage/set size are measured on the fixed evaluation
set. Both summary statistics (mean/std) AND the raw per-draw values
are saved - the raw values are required for any downstream
distributional analysis (e.g. comparison against the theoretical
Beta-distributed calibration-conditional coverage reference), which
cannot be done from summary statistics alone.

Note on independence: all 20 draws at a given N share the SAME fixed
500-example evaluation set. This means the 20 coverage observations
are not fully independent samples of the unconditional coverage
distribution - they are repeated measurements against one fixed
target. Any distributional comparison using these raw draws (e.g. a
KS test against the theoretical Beta reference) must treat this as a
descriptive comparison, not a formal independence-assuming hypothesis
test. See the analysis script and dissertation Methodology section for
the full caveat.

Usage:
    python run_recalibration_sweep.py --dataset imagenet_r --method aps
    python run_recalibration_sweep.py --dataset imagenet_a --method lac
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.conformal.factory import METHOD_CLASSES as _METHOD_CLASSES
from src.embeddings.cache import EmbeddingCache
from src.metrics.coverage import coverage_report
from src.utils.config import load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

_N_BUDGETS = [10, 25, 50, 100, 250]
_NUM_DRAWS = 20
_RECAL_POOL_SIZE = 500
_EVAL_SET_SIZE = 500


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _load_shift_probs_and_labels(
    dataset_name: str, model_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load cached test embeddings for the given shift dataset and
    reconstruct class probabilities. Requires that EXP003/EXP004 (or
    an equivalent run_shift_eval.py call) has already cached these
    embeddings - this script does not stream data or run CLIP itself.
    """
    from src.datasets.imagenet_class_names import load_imagenet_class_names
    from src.models.manager import ModelManager
    import src.models.clip_model  # noqa: F401

    cache = EmbeddingCache()
    cache_key = f"{model_name}_{dataset_name}_test"

    if not cache.exists(cache_key):
        raise FileNotFoundError(
            f"No cached embeddings found for '{cache_key}'. Run "
            f"run_shift_eval.py --dataset {dataset_name} first."
        )

    embeddings, labels = cache.load(cache_key)

    model = ModelManager(model_name)
    model.load()

    full_class_names = load_imagenet_class_names()

    if dataset_name == "imagenet_r":
        from src.datasets.imagenet_r_class_mapping import (
            load_imagenet_r_local_to_full_mapping,
        )

        mapping = load_imagenet_r_local_to_full_mapping()
        active_class_names = [full_class_names[mapping[i]] for i in range(len(mapping))]
    elif dataset_name == "imagenet_a":
        from src.datasets.imagenet_a_class_mapping import (
            load_imagenet_a_local_to_full_mapping,
        )

        mapping = load_imagenet_a_local_to_full_mapping()
        active_class_names = [full_class_names[mapping[i]] for i in range(len(mapping))]
    else:
        raise ValueError(
            f"Recalibration sweep only supports 'imagenet_r' or 'imagenet_a', "
            f"got '{dataset_name}'."
        )

    text_embeddings = model.encode_text(active_class_names).cpu().numpy()
    probs = _softmax(model.logit_scale * (embeddings @ text_embeddings.T))
    return probs, labels


def _run_sweep_for_dataset(
    probs: np.ndarray, labels: np.ndarray, method_name: str, alpha: float, seed: int
) -> dict:
    """
    Execute the full N-budget Monte Carlo sweep for one dataset and
    one conformal method, using a fixed recalibration pool / evaluation
    set split.
    """
    rng = np.random.default_rng(seed)

    total_available = probs.shape[0]
    required = _RECAL_POOL_SIZE + _EVAL_SET_SIZE
    if total_available < required:
        raise ValueError(
            f"Need at least {required} cached examples for the sweep "
            f"(pool={_RECAL_POOL_SIZE} + eval={_EVAL_SET_SIZE}), "
            f"but only {total_available} are cached."
        )

    # Fixed split, drawn once, held constant for the entire sweep.
    all_indices = rng.permutation(total_available)
    pool_indices = all_indices[:_RECAL_POOL_SIZE]
    eval_indices = all_indices[_RECAL_POOL_SIZE : _RECAL_POOL_SIZE + _EVAL_SET_SIZE]

    pool_probs, pool_labels = probs[pool_indices], labels[pool_indices]
    eval_probs, eval_labels = probs[eval_indices], labels[eval_indices]

    method_class = _METHOD_CLASSES[method_name]
    results_by_n = {}

    for n in _N_BUDGETS:
        draw_coverages = []
        draw_set_sizes = []

        for draw in range(_NUM_DRAWS):
            draw_rng = np.random.default_rng(seed + n * 1000 + draw)
            sample_indices = draw_rng.choice(_RECAL_POOL_SIZE, size=n, replace=False)

            calibration_probs = pool_probs[sample_indices]
            calibration_labels = pool_labels[sample_indices]

            method = method_class(alpha=alpha)
            method.calibrate(calibration_probs, calibration_labels)
            prediction_sets = method.predict_sets(eval_probs)

            report = coverage_report(prediction_sets, eval_labels, alpha=alpha)
            draw_coverages.append(report["empirical_coverage"])
            draw_set_sizes.append(report["average_set_size"])

        results_by_n[n] = {
            "coverage_mean": float(np.mean(draw_coverages)),
            "coverage_std": float(np.std(draw_coverages)),
            "set_size_mean": float(np.mean(draw_set_sizes)),
            "set_size_std": float(np.std(draw_set_sizes)),
            "coverage_draws": [float(c) for c in draw_coverages],
            "set_size_draws": [float(s) for s in draw_set_sizes],
            "num_draws": _NUM_DRAWS,
        }

        logger.info(
            f"N={n}: coverage={results_by_n[n]['coverage_mean']:.4f} "
            f"+/- {results_by_n[n]['coverage_std']:.4f}, "
            f"set_size={results_by_n[n]['set_size_mean']:.2f} "
            f"+/- {results_by_n[n]['set_size_std']:.2f}"
        )

    return results_by_n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo target-domain recalibration budget sweep."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["imagenet_r", "imagenet_a"]
    )
    parser.add_argument(
        "--method", type=str, required=True, choices=sorted(_METHOD_CLASSES.keys())
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    set_seed(config.seed.value)

    logger.info(f"Loading cached probabilities for '{args.dataset}'...")
    probs, labels = _load_shift_probs_and_labels(args.dataset, config.model.name)

    logger.info(
        f"Starting recalibration sweep: dataset={args.dataset}, method={args.method}, "
        f"alpha={config.calibration.alpha}, N budgets={_N_BUDGETS}, draws per N={_NUM_DRAWS}"
    )

    results = _run_sweep_for_dataset(
        probs, labels, args.method, config.calibration.alpha, config.seed.value
    )

    output_dir = Path("results") / f"RECAL-{args.dataset}-{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "dataset": args.dataset,
        "method": args.method,
        "alpha": config.calibration.alpha,
        "target_coverage": 1 - config.calibration.alpha,
        "recal_pool_size": _RECAL_POOL_SIZE,
        "eval_set_size": _EVAL_SET_SIZE,
        "num_draws": _NUM_DRAWS,
        "results_by_n": results,
    }

    with open(output_dir / "recovery.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n--- Recalibration Recovery Sweep: {args.dataset} / {args.method} ---")
    print(f"{'N':>6} {'Coverage':>18} {'Set Size':>18}")
    for n in _N_BUDGETS:
        r = results[n]
        print(
            f"{n:>6} {r['coverage_mean']:.4f} +/- {r['coverage_std']:.4f}   "
            f"{r['set_size_mean']:.2f} +/- {r['set_size_std']:.2f}"
        )

    logger.info(f"Results archived to {output_dir / 'recovery.json'}")


if __name__ == "__main__":
    main()