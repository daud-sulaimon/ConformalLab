# ConformalLab / ReCal-CP

**ConformalLab** is the research codebase for the MSc dissertation *Target-Free Conformal Adaptation for
Zero-Shot Vision–Language Models
Under Distribution Shift* (Daud Sulaimon). 
It implements and compares four strategies for restoring conformal prediction reliability when a zero-shot CLIP model is deployed under distribution shift: frozen source calibration, pseudo-calibration, confidence-filtered pseudo-calibration, entropy-scaled calibration (ECP), and labelled target-domain recalibration.

**ReCal-CP** is the model-agnostic evaluation core (`src/conformal/core.py`) this comparison is built on — verified to contain no CLIP or ImageNet dependency, so the same evaluation logic works on any classifier's probability/logit outputs.

---

## The research question

When a source-calibrated conformal predictor is deployed on shifted target data and target labels are unavailable, which substitute — model-generated labels, confidence-filtered labels, or unlabelled predictive uncertainty — actually restores reliable coverage, and how does each compare against having real target labels?

**Headline finding**: at the same zero-label cost, outcomes on ImageNet-A range from 28.3% to 91.7% coverage (nominal target: 90%) depending entirely on *which* target information is extracted from the model. Confidence-based pseudo-label filtering — the standard mitigation proposed in the label-free calibration literature — helps adaptive conformal scores (APS, RAPS) and actively harms the simple LAC score, an effect isolated from sample-size confounds by two independent matched-N controls and explained analytically: for hard pseudo-labels, every score tested is a deterministic function of the model's top-1 confidence, with opposite sign for LAC versus APS/RAPS.

---

## Quick start

```bash
conda create -n conformallab python=3.11
conda activate conformallab
pip install -r requirements.txt
hf auth login   # required: ILSVRC/imagenet-1k is a gated dataset
```

```bash
python demo_recal_cp.py --dataset imagenet_a --method aps
```

Runs in under a minute from cached embeddings (no new model inference), printing a frozen-threshold diagnostic and a recalibration-budget sweep, and saving two figures plus JSON/CSV output.

---

## What's actually in this repository

| Experiment | Script | What it tests |
|---|---|---|
| Baseline degradation | `run_split_cp.py`, `run_shift_eval.py` | Frozen-threshold coverage across ImageNet/V2/R/A |
| Randomisation sensitivity | `run_randomization_sensitivity*.py` | Deterministic vs. literature-specified randomised APS/RAPS scores (found up to 18.3pp difference — see Limitations) |
| C3: Pseudo-calibration | `run_pseudo_calibration.py` | Calibrating on the model's own top-1 predictions instead of true labels |
| C4: Confidence filtering | `run_confidence_filtered_pseudo_calibration.py`, `run_c4_matched_n_control.py`, `run_c4_random_pseudo.py` | Whether filtering pseudo-labels by confidence helps calibration, isolated from sample-size effects by two matched-N controls |
| C5: Entropy-scaled calibration (ECP) | `run_ecp.py` | Rescaling a source threshold using unlabelled target entropy (adapted from Kasa et al., 2025, to LAC's negatively-oriented score — derivation in `tests/test_ecp_mapping.py`) |
| Cross-regime comparison | `run_information_regime_comparison.py`, `run_master_table_imagenet_a.py` | All strategies on one protocol with one pre-registered criterion |

Every experiment reads from cached CLIP embeddings (`embeddings/*.npy`) — none requires GPU access or re-running model inference.

---

## The core API (model-agnostic)

```python
from src.conformal.core import baseline_evaluation, recovery_sweep

# Frozen-threshold diagnostic
report = baseline_evaluation(
    source_probs, source_labels,
    target_probs, target_labels,
    method="aps",       # "lac", "aps", or "raps"
    alpha=0.1,
)

# Labelled recalibration budget sweep
result = recovery_sweep(
    target_probs=target_probs,     # or target_logits=raw_logits
    target_labels=target_labels,
    method="aps",
    budgets=(10, 25, 50, 100, 250),
    draws=20,
)
```

`randomize=True` is the default for APS/RAPS — this is the literature-specified score formulation, not the deterministic simplification this project's own sensitivity check found to inflate results (see Limitations). No CLIP, ImageNet, or Hugging Face import exists anywhere in `src/conformal/core.py`, verified by `tests/test_core_portability.py` against purely synthetic data.

---

## Interpreting the results

For any recalibration-budget output:

- **Coverage is not monotonic in N** — it typically peaks near N=25 (an artefact of conservative small-sample quantile estimation), not at the largest tested budget. Consider the whole curve, not one point.
- **The pre-registered recovery criterion** (Section 4.3 of the dissertation): coverage within ±2 percentage points of the 90% target **and** standard deviation ≤ 0.03 across repeated draws, both required simultaneously, fixed before any result was examined. "Criterion not met" is reported as a real, informative outcome, not a failure of the tool.
- **Confidence filtering is not a universal fix** — check which conformal method you're using before applying it. See the C4 experiments and the dissertation's Section 5.4/5.4.1 for the score-dependent mechanism.

---

## Limitations (see dissertation Chapter 6/Limitations for the full list)

- **APS and RAPS default to the randomised score variant.** An earlier implementation used a deterministic simplification that was found, via a 5-seed sensitivity check across all four datasets, to inflate coverage and set size by up to 18.3 percentage points — most severely under cross-label-space threshold transfer (ImageNet → ImageNet-A). This is reported as a methodological finding in its own right, not only a correction: it demonstrates that benchmark conclusions in this literature can depend materially on an implementation detail rarely reported in published work. Guarded against recurring by `tests/test_core_randomize_default.py`.
- **ECP is implemented for LAC only.** APS and RAPS score classes by cumulative rank-ordered probability mass, not a per-class likelihood, and Kasa et al. (2025) do not specify a correspondence between their positively-oriented formulation and a cumulative-sum score. Extending without a derived mapping was judged a research risk after two earlier instances of exactly this kind of unattributed-mapping error during development; it is left as an explicit open problem.
- **ECP's correction does not transfer across shift severity.** The quadratic scaling variant that satisfies the recovery criterion on ImageNet-A (severe shift) overcorrects on ImageNet-R and ImageNet-V2 (mild shift), inflating set sizes 4–5× above baseline. A fixed scaling rule is not adequate across the shift continuum tested.
- **No claim of generalisation beyond CLIP ViT-B/32** on the ImageNet/V2/R/A continuum. All findings are scoped to this model and this shift family.
- **A class-grouped data-streaming bug was found and corrected** during development: WebDataset shards for ImageNet-R/A/V2 are written class-sequentially, and streaming without shuffling produced severely non-representative samples (1,000 nominal ImageNet-R examples spanning only 6 of 200 classes) — a silent violation of the exchangeability assumption underlying every downstream result, with no error raised. Fixed via `stream.shuffle(seed=..., buffer_size=50000)` in each dataset loader before any dissertation result was finalised.

---

## Class-space handling for ImageNet-R and ImageNet-A

Both cover only 200 of ImageNet's 1,000 classes. Softmax is renormalised over the relevant 200-class subset using Hendrycks et al.'s own published index mappings (`src/datasets/imagenet_r_class_mapping.py`, `imagenet_a_class_mapping.py`), not an independently constructed correspondence.

---

## Repository structure

```
src/
  conformal/      BaseConformalMethod, LAC/APS/RAPS (deterministic + randomised),
                   core.py (model-agnostic API), report.py, factory.py
  datasets/       Dataset loaders + class-index mappings (ImageNet/V2/R/A)
  models/         CLIP wrapper
  embeddings/     Embedding cache + extraction pipeline
  metrics/        Coverage reporting + recovery-criterion computation
  utils/          Logging, seeding, config loading
tests/            118 tests: per-method statistical checks, model-agnostic
                   portability, ECP mapping verification, provenance test
results/          Every experiment's archived JSON/TXT output and figures
demo_recal_cp.py                          One-command demonstration
run.py                                    EXP001: zero-shot baseline accuracy
run_split_cp.py / run_shift_eval.py       Frozen-threshold calibration + evaluation
run_recalibration_sweep.py                Labelled recalibration budget sweep
run_pseudo_calibration.py                 C3
run_confidence_filtered_pseudo_calibration.py, run_c4_*.py   C4 + matched-N controls
run_ecp.py                                C5
run_information_regime_comparison.py, run_master_table_imagenet_a.py   Cross-regime comparison
```

---

## Reproducibility

Every experiment is seeded (`src/utils/seed.py`, base seed 42), with repeated draws given deterministic sub-seeds so results are exactly reproducible while genuinely sampling different partitions. All configuration, raw output, and summary statistics are archived under `results/` — no reported number in the dissertation lacks a corresponding file in this repository.

```bash
pytest -v
```

118 tests, including `tests/test_ecp_mapping.py` (verifies the ECP-to-LAC derivation before any ECP result is trusted) and `tests/test_core_randomize_default.py` (guards against the artefact silently reverting to the superseded deterministic score variant).

---



MIT. See `LICENSE`.
