# tests/test_report.py
import numpy as np
from src.conformal.core import baseline_evaluation, recovery_sweep
from src.conformal.report import print_diagnostic_report, save_report, plot_recovery_curve, plot_coverage_efficiency


def _synthetic(n, k, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    probs = rng.uniform(0.01, 0.1, size=(n, k))
    probs[np.arange(n), labels] += rng.uniform(0.3, 0.6, size=n)
    return probs / probs.sum(axis=1, keepdims=True), labels


def test_report_pipeline_end_to_end(tmp_path):
    src_p, src_y = _synthetic(500, 10, 0)
    tgt_p, tgt_y = _synthetic(500, 10, 1)

    baseline = baseline_evaluation(src_p, src_y, tgt_p, tgt_y, method="aps")
    sweep = recovery_sweep(target_probs=tgt_p, target_labels=tgt_y, method="aps", budgets=[10, 50], draws=5)

    print_diagnostic_report(baseline, sweep)
    save_report(baseline, sweep, tmp_path)
    plot_recovery_curve(sweep, tmp_path / "fig1.png")
    plot_coverage_efficiency(sweep, tmp_path / "fig2.png")

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "recovery_table.csv").exists()
    assert (tmp_path / "fig1.png").exists()
    assert (tmp_path / "fig2.png").exists()