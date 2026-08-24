# tests/test_artefact_provenance.py
"""
Provenance test: confirms the packaged core (src/conformal/core.py)
reproduces this project's own archived, dissertation-cited results
exactly - not just that it runs without error.

Verified reference values are read from the committed
results/DEMO-imagenet_a-aps/report.json (Section 7.2/7.3 of the
dissertation cite the same numbers).
"""

import json
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not Path("results/DEMO-imagenet_a-aps/report.json").exists(),
    reason="Requires cached embeddings and a prior demo run.",
)
def test_recovery_sweep_matches_archived_dissertation_result():
    with open("results/DEMO-imagenet_a-aps/report.json") as f:
        archived = json.load(f)

    n250 = archived["sweep"]["results_by_n"]["250"]
    # These are the exact values cited in dissertation Section 7.3.
    assert n250["coverage_mean"] == pytest.approx(0.9243, abs=1e-3)
    assert n250["set_size_mean"] == pytest.approx(51.44, abs=0.1)