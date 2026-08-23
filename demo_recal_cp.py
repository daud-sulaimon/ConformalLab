"""
ReCal-CP viva demonstration script.

Loads already-cached CLIP/ImageNet embeddings (no new inference), runs
the frozen-threshold diagnostic and the recovery-budget sweep through
the model-agnostic src.conformal.core API, and produces the printed
report plus two figures - the single command that demonstrates the
whole artefact end to end.

This script IS a "research adapter" (it knows about CLIP/ImageNet/
Hugging Face) that feeds arrays into the model-agnostic core - it is
not itself the proof of model-agnosticism. That proof is
tests/test_core_portability.py, which contains no CLIP/ImageNet import
at all. This script exists to show the same core engine running on
real research data.

Usage:
    python demo_recal_cp.py --dataset imagenet_a --method aps
    python demo_recal_cp.py --dataset imagenet_r --method lac
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.conformal.core import baseline_evaluation, recovery_sweep
from src.conformal.report import (
    plot_coverage_efficiency,
    plot_recovery_curve,
    print_diagnostic_report,
    save_report,
)
from src.datasets.imagenet_class_names import load_imagenet_class_names
from src.embeddings.cache import EmbeddingCache
from src.models.manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import configure_logging, get_logger
from src.utils.seed import set_seed

import src.models.clip_model  # noqa: F401

logger = get_logger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _get_active_class_names(dataset: str, full_class_names: list) -> list:
    if dataset == "imagenet_r":
        from src.datasets.imagenet_r_class_mapping import load_imagenet_r_local_to_full_mapping
        mapping = load_imagenet_r_local_to_full_mapping()
    elif dataset == "imagenet_a":
        from src.datasets.imagenet_a_class_mapping import load_imagenet_a_local_to_full_mapping
        mapping = load_imagenet_a_local_to_full_mapping()
    else:
        raise ValueError(f"Demo supports 'imagenet_r' or 'imagenet_a', got '{dataset}'.")
    return [full_class_names[mapping[i]] for i in range(len(mapping))]


def main() -> None:
    parser = argparse.ArgumentParser(description="ReCal-CP viva demonstration.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["imagenet_r", "imagenet_a"])
    parser.add_argument("--method", type=str, required=True, choices=["lac", "aps", "raps"])
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    set_seed(config.seed.value)

    print(f"\nLoading cached embeddings for '{args.dataset}' (no new CLIP inference)...")

    model = ModelManager(config.model.name)
    model.load()

    full_class_names = load_imagenet_class_names()
    active_class_names = _get_active_class_names(args.dataset, full_class_names)

    text_full = model.encode_text(full_class_names).cpu().numpy()
    text_active = model.encode_text(active_class_names).cpu().numpy()

    cache = EmbeddingCache()

    source_embeddings, source_labels = cache.load(f"{config.model.name}_imagenet_calibration")
    source_probs = _softmax(model.logit_scale * (source_embeddings @ text_full.T))

    target_embeddings, target_labels = cache.load(f"{config.model.name}_{args.dataset}_test")
    target_probs = _softmax(model.logit_scale * (target_embeddings @ text_active.T))

    # --- Stage 1: frozen-threshold diagnostic (Component B) ---
    # Note: this reuses the identical frozen-threshold transfer protocol
    # as run_shift_eval.py, so it correctly reproduces this dissertation's
    # Section 7.2 numbers.
    baseline = baseline_evaluation(
        source_probs, source_labels, target_probs, target_labels,
        method=args.method, alpha=config.calibration.alpha,
    )

    # --- Stage 2: recovery-budget sweep (Component C) ---
    sweep = recovery_sweep(
        target_probs=target_probs, target_labels=target_labels,
        method=args.method, alpha=config.calibration.alpha,
        budgets=(10, 25, 50, 100, 250), draws=20,
        recal_pool_size=500, eval_set_size=500, seed=config.seed.value,
    )

    # --- Stage 3: report (Component D) ---
    print_diagnostic_report(baseline, sweep)

    output_dir = Path("results") / f"DEMO-{args.dataset}-{args.method}"
    save_report(baseline, sweep, output_dir)
    plot_recovery_curve(sweep, output_dir / "recovery_curve.png")
    plot_coverage_efficiency(sweep, output_dir / "coverage_efficiency.png")

    print(f"\nFull report, table, and figures saved to {output_dir}/")


if __name__ == "__main__":
    main()