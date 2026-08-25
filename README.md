# ReCal-CP / ConformalLab

**ReCal-CP** is a model-agnostic calibration-resource diagnostic tool for conformal prediction under distribution shift. It is the software artefact for the MSc dissertation *Target-Domain Calibration Budgets for Reliable Conformal Prediction in Zero-Shot Vision-Language Models Under Distribution Shift* (Daud Sulaimon).

**ConformalLab** is the underlying research codebase — dataset loaders, model wrappers, embedding caching, and the conformal methods themselves — that ReCal-CP's core engine is built on. ReCal-CP is the specific, reusable, model-agnostic packaging of that engine's results.

---

## What ReCal-CP Is

Given a classifier's calibration-set and target-set outputs, ReCal-CP answers two questions a practitioner actually has when deploying under distribution shift:

1. **Frozen-threshold diagnostic** — if I calibrate once on my source data and deploy unchanged, how far short of my target coverage do I actually fall?
2. **Recalibration-budget sweep** — how does coverage respond as the target calibration budget increases, and does any tested budget satisfy the predefined recovery criterion?

It implements three conformal methods (LAC/Split CP, APS, RAPS) sharing one interface, and reports not just a single number but the full recovery curve with variance, so both outcomes are supported honestly:

N* = 50
or
N* = NOT REACHED WITHIN TESTED RANGE


**It is not a new conformal algorithm.** Existing libraries (MAPIE, TorchCP, PUNCC, Crepes) implement the algorithms. ReCal-CP's contribution is the calibration-resource workflow those libraries don't provide.

---

## Quick Start — the 5-minute demo workflow

```bash
conda activate conformallab
python demo_recal_cp.py --dataset imagenet_a --method aps
```

This is ReCal-CP's primary entry point and reproduces the viva demonstration end to end, in under a minute, using already-cached embeddings (no new model inference):

demo_recal_cp.py
|
Frozen coverage diagnostic
|
N = 10, 25, 50, 100, 250
|
coverage + SD + set size
|
N* / NOT REACHED
|
two plots + JSON + CSV


Console output shows the frozen diagnostic and the full recovery table; `results/DEMO-imagenet_a-aps/` receives `report.json`, `recovery_table.csv`, and two figures (recovery curve, coverage-vs-efficiency).

---

## Input Format

The core engine (`src/conformal/core.py`) is model-agnostic: it has no CLIP, ImageNet, or Hugging Face dependency anywhere in the file (verified by `tests/test_core_portability.py`, which runs it against purely synthetic data).

```python
from src.conformal.core import baseline_evaluation, recovery_sweep

# Frozen-threshold diagnostic
report = baseline_evaluation(
    source_probs, source_labels,   # calibration data: (n_source, n_classes), (n_source,)
    target_probs, target_labels,   # target data: (n_target, n_classes), (n_target,)
    method="aps",                  # "lac", "aps", or "raps"
    alpha=0.1,                     # miscoverage rate
)

# Recalibration-budget sweep
result = recovery_sweep(
    target_probs=target_probs,     # OR target_logits=raw_logits
    target_labels=target_labels,
    method="aps",
    budgets=(10, 25, 50, 100, 250),
    draws=20,
)
```

**`baseline_evaluation()` accepts probabilities only.** **`recovery_sweep()` accepts either probabilities or raw logits** (via `target_probs=` or `target_logits=`; softmax normalisation happens in exactly one documented location inside the function when logits are given). This asymmetry is intentional, not an oversight — `baseline_evaluation()` was never used with a logits input in this project's experiments, and adding a second, untested logits path there would add surface area without adding verified value. If you have logits for `baseline_evaluation()`, normalise them yourself first (`exp(x) / exp(x).sum()` per row).

`randomize=True` is the default for APS/RAPS (see Limitations below for why this matters) and is silently ignored for LAC, which has no randomised variant.

---

## Reproducing a Dissertation Result

```bash
python demo_recal_cp.py --dataset imagenet_a --method aps
```

This closely reproduces the figures cited in dissertation Sections 7.2–7.3: frozen coverage in the 70–72% range (seed-dependent, documented in Section 8.4), and N=250 recalibrated coverage of 92.4% with mean set size 51.44. The match is checked automatically by `tests/test_artefact_provenance.py`, within a small numerical tolerance (coverage within ±0.001, set size within ±0.1) rather than exact equality — this tolerance is deliberate, since floating-point behaviour can vary subtly across platforms and library versions even with identical seeding, and claiming exact bit-for-bit reproduction would overstate what the test actually verifies.

To reproduce ImageNet-R instead (a milder shift, where recalibration succeeds more completely):

```bash
python demo_recal_cp.py --dataset imagenet_r --method lac
```

---

## Interpreting the Output

Frozen coverage: 70.7% <- coverage BEFORE any recalibration
Nominal coverage: 90.0% <- your target
Coverage gap: -19.3% <- how far short the frozen threshold falls

 N   Mean coverage       SD   Mean set size
10          88.3%   0.0894           50.40   <- small N: noisy, unreliable
25          98.1%   0.0132           98.74   <- small-sample quantile conservatism


250 92.4% 0.0105 51.44 <- largest tested budget

N*: NOT REACHED


- **Frozen coverage / gap**: what happens if you never recalibrate. Large negative gaps indicate the shift is severe enough that source calibration alone is unsafe.
- **The N-table**: coverage may be non-monotonic across tested budgets — often, though not always, showing higher and noisier values at small N, reflecting that the calibrated quantile is estimated from few target examples there. The complete curve should be considered rather than interpreting a single N in isolation; this non-monotonicity is itself part of what the sweep is designed to surface, not something to discount (dissertation Section 7.3).

- **N\***: the smallest tested budget where coverage lands within ±2 percentage points of target AND standard deviation ≤ 0.03 (both criteria required simultaneously, fixed in advance — dissertation Section 4.5.1). **"NOT REACHED" is a real, informative result** — it means no tested budget was sufficient, not that the tool failed. It is reported honestly rather than dressed up as a number.

---

## Limitations

- **Evaluation batch is finite (500 examples in this project's experiments)**, not an infinite population — reported coverage has genuine statistical noise from this, on top of calibration-sample variance. See dissertation Section 7.4 for the finite-batch (Beta-Binomial) statistical treatment.
- **The 20 repeated draws per calibration budget share one fixed evaluation set** — they are not fully independent samples. Documented in dissertation Section 4.5.2.
- **`randomize=True` is the literature-correct default for APS/RAPS**, but this project's very first implementation defaulted silently to a deterministic variant that materially inflates coverage and set size (dissertation Section 8.4) — a bug we found and fixed, guarded against recurring by `tests/test_core_randomize_default.py`.
- **No claim of generalisation beyond this dissertation's evaluated model (CLIP ViT-B/32), datasets (ImageNet/V2/R/A), and calibration-budget range (N ≤ 250)** — see dissertation Section 10 for the full, honest limitations list, including a documented instance where an earlier draft's finding could not be reconciled with the underlying data and was withdrawn.

---

## Repository Structure

src/
conformal/ BaseConformalMethod, LAC/APS/RAPS, core.py (model-agnostic API), report.py
datasets/ BaseDataset + ImageNet / ImageNet-V2 / ImageNet-R / ImageNet-A loaders
models/ BaseModel + CLIPModel wrapper
embeddings/ embedding cache + extraction pipeline
metrics/ coverage reporting + N* recovery-criterion computation
utils/ logging, seeding, config loading
tests/ covers every module above, including artefact-layer portability/provenance tests
results/ archived experiment outputs (coverage.json, threshold.json, recovery.json)
demo_recal_cp.py the viva demonstration entry point
run.py EXP001: zero-shot CLIP baseline accuracy
run_split_cp.py calibrate a conformal method (--method lac|aps|raps) on ImageNet
run_shift_eval.py evaluate a frozen threshold on a shift dataset
run_recalibration_sweep.py Monte Carlo N-budget recalibration sweep (the dissertation's RQ3 experiment)


---

## Setup

```bash
conda create -n conformallab python=3.11
conda activate conformallab
pip install -r requirements.txt
```

Requires a Hugging Face account with access granted to `ILSVRC/imagenet-1k` (gated dataset — accept its terms on the dataset page first), then:

```bash
hf auth login
```

---

## Reproducing the Full Experiment Set

**1. Baseline zero-shot accuracy (EXP001)**
```bash
python run.py --config configs/default.yaml
```

**2. Calibrate each conformal method on ImageNet**
```bash
python run_split_cp.py --config configs/default.yaml --method lac
python run_split_cp.py --config configs/default.yaml --method aps
python run_split_cp.py --config configs/default.yaml --method raps
```

**3. Evaluate each method's frozen threshold on every shift dataset**
```bash
for method in lac aps raps; do
  for dataset in imagenet_v2 imagenet_r imagenet_a; do
    python run_shift_eval.py --config configs/default.yaml --dataset $dataset --method $method
  done
done
```

**4. Run the recalibration budget sweep**
```bash
for dataset in imagenet_r imagenet_a; do
  for method in lac aps raps; do
    python run_recalibration_sweep.py --dataset $dataset --method $method
  done
done
```

**5. Compute N\* and inspect distributional verification**
```bash
python -m scripts.compute_all_n_star
```

Plots comparing observed recalibration draws against the theoretical Beta-Binomial reference are in `results/distributional_analysis/`.

---

## Class-Space Handling for ImageNet-R and ImageNet-A

ImageNet-R and ImageNet-A each cover only 200 of ImageNet's 1000 classes. Following standard practice in the calibration/robustness literature, evaluation on these datasets restricts the model's output to the relevant 200-class subset before computing softmax, rather than the full 1000. The exact 200-class mapping for each dataset is sourced from the datasets' original authors (Hendrycks et al.'s published index lists), not independently invented — see `src/datasets/imagenet_r_class_mapping.py` and `src/datasets/imagenet_a_class_mapping.py`.

---

## A Known, Corrected Data-Loading Pitfall

An earlier version of this codebase streamed ImageNet-R/A/V2 from their Hugging Face WebDataset mirrors without shuffling. Because WebDataset shards are written class-sequentially, this produced severely class-grouped samples — 1,000 nominally sampled ImageNet-R examples spanning only 6 of 200 classes — a silent violation of the exchangeability assumption underlying every downstream conformal result, with no error raised. This was caught, diagnosed, and fixed (`stream.shuffle(seed=..., buffer_size=50000)` in each dataset loader) before any dissertation result was finalised. It is documented here, and in the dissertation's Limitations section, because it connects a mundane implementation choice directly to the exchangeability assumption the entire experiment depends on.

---

## Reproducibility

Every experiment run is seeded (`src/utils/seed.py`, default seed 42) and archives its exact config, results, and (where applicable) frozen calibration threshold under `results/`. The baseline experiment is additionally tagged at git commit `exp001-baseline`.

---

## Testing

```bash
pytest -v
```

Covers every module including statistical checks that empirical coverage approximates its target for each conformal method, and reproducibility checks on the recalibration sweep. Two tests specific to the artefact-packaging layer are particularly important, since packaging itself introduced a real regression during development (see Limitations):

- **`tests/test_core_randomize_default.py`** — guards against `baseline_evaluation()`/`recovery_sweep()` silently defaulting to the superseded deterministic APS/RAPS variant, which is exactly the bug this test was written to catch after it happened once.
- **`tests/test_artefact_provenance.py`** — confirms the packaged core reproduces this project's own archived, dissertation-cited results within a stated numerical tolerance, not merely that the code runs without error.

---

## License

MIT. See `LICENSE`.