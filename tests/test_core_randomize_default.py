# tests/test_core_randomize_default.py
import numpy as np
from src.conformal.core import baseline_evaluation


def test_baseline_evaluation_defaults_to_randomized_for_aps():
    """Guards against silently reproducing this project's superseded
    deterministic results (Section 8.4)."""
    rng = np.random.default_rng(0)
    src_p = rng.dirichlet(np.ones(20), size=500)
    src_y = rng.integers(0, 20, size=500)
    tgt_p = rng.dirichlet(np.ones(20), size=500)
    tgt_y = rng.integers(0, 20, size=500)

    default_report = baseline_evaluation(src_p, src_y, tgt_p, tgt_y, method="aps")
    deterministic_report = baseline_evaluation(src_p, src_y, tgt_p, tgt_y, method="aps", randomize=False)

    assert default_report["randomize"] is True
    # Deterministic variant is documented as inflating set size - if the
    # default silently used it, this assertion would catch that.
    assert default_report["average_set_size"] < deterministic_report["average_set_size"]