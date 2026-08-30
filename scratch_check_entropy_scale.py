"""
Scratch diagnostic: check the magnitude of ECP's entropy quantile
u_test on real cached data, before choosing the uncertainty scaling
function f(.).

Kasa et al. (2025), Algorithm 1:
    u_test <- q_{1-alpha}({h(x_i)}_{i in [N]})
    u_test <- f(u_test)
    C(x_i) := {y : s(x_i,y) * max(1, u_test) >= tau_D}

The paper validates two choices of f: linear (ECP1, f(u) = u) and
quadratic (ECP2, f(u) = u^2). Which is sensible depends entirely on
the magnitude of u_test in nats, which depends on the number of
classes: max possible entropy is log(K), so ~6.9 for K=1000 and ~5.3
for K=200.

If u_test is around 1-2, both linear and quadratic are plausible
scale factors. If u_test is near 5, quadratic gives a ~25x threshold
inflation, which would saturate every prediction set and make the
method meaningless in this setting - worth knowing before building
the full experiment around it.

Run once, read the output, then delete.

Usage:
    python scratch_check_entropy_scale.py
"""

from __future__ import annotations

import logging

import numpy as np

from src.datasets.imagenet_class_names import load_imagenet_class_names
from src.embeddings.cache import EmbeddingCache
from src.models.manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import configure_logging

import src.models.clip_model  # noqa: F401

ALPHA = 0.1


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row, in nats, matching Kasa et al. Eq. 5."""
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def _load_probs(dataset: str, model, full_class_names, cache):
    if dataset in ("imagenet_r", "imagenet_a"):
        if dataset == "imagenet_r":
            from src.datasets.imagenet_r_class_mapping import (
                load_imagenet_r_local_to_full_mapping as load_map,
            )
        else:
            from src.datasets.imagenet_a_class_mapping import (
                load_imagenet_a_local_to_full_mapping as load_map,
            )
        mapping = load_map()
        class_names = [full_class_names[mapping[i]] for i in range(len(mapping))]
        embeddings, labels = cache.load(f"clip_{dataset}_test")
    elif dataset == "imagenet_v2":
        class_names = full_class_names
        embeddings, labels = cache.load("clip_imagenet_v2_test")
    else:
        class_names = full_class_names
        cal_emb, cal_lab = cache.load("clip_imagenet_calibration")
        test_emb, test_lab = cache.load("clip_imagenet_test")
        embeddings = np.concatenate([cal_emb, test_emb], axis=0)
        labels = np.concatenate([cal_lab, test_lab], axis=0)

    text_emb = model.encode_text(class_names).cpu().numpy()
    probs = _softmax(model.logit_scale * (embeddings @ text_emb.T))
    return probs, labels, len(class_names)


def main() -> None:
    configure_logging()
    logging.getLogger("conformallab").setLevel(logging.WARNING)

    config = load_config("configs/default.yaml")
    model = ModelManager("clip")
    model.load()
    full_class_names = load_imagenet_class_names()
    cache = EmbeddingCache()

    print(
        f"{'dataset':>13} {'K':>6} {'max_H':>7} {'mean_H':>8} "
        f"{'u_test':>8} {'f=u':>7} {'f=u^2':>8} {'frac_max':>9}"
    )
    print("-" * 78)

    for dataset in ["imagenet", "imagenet_v2", "imagenet_r", "imagenet_a"]:
        probs, labels, n_classes = _load_probs(
            dataset, model, full_class_names, cache
        )
        ent = _entropy(probs)
        u_test = float(np.quantile(ent, 1 - ALPHA))  # beta = 1 - alpha per paper
        max_entropy = float(np.log(n_classes))

        linear_scale = max(1.0, u_test)
        quadratic_scale = max(1.0, u_test ** 2)

        print(
            f"{dataset:>13} {n_classes:>6} {max_entropy:>7.3f} {ent.mean():>8.3f} "
            f"{u_test:>8.3f} {linear_scale:>7.3f} {quadratic_scale:>8.3f} "
            f"{u_test / max_entropy:>9.3f}"
        )

    print(
        "\nu_test = the (1-alpha)-quantile of per-example prediction entropy,\n"
        "matching Kasa et al.'s beta = 1 - alpha default.\n"
        "f=u is ECP1 (linear scaling); f=u^2 is ECP2 (quadratic scaling).\n"
        "frac_max = u_test / log(K), i.e. how close the entropy quantile sits\n"
        "to the theoretical maximum for that label space."
    )


if __name__ == "__main__":
    main()