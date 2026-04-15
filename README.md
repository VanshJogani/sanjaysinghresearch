# Adaptive Fairness Unlearning (AFU)

A Python library that continuously monitors a deployed model's fairness in a streaming setting, detects bias re-emergence, identifies harmful data points via influence functions, and selectively unlearns them — without full retraining.

---

## Overview

Deployed models can become unfair over time as data distributions shift or biased batches arrive. Full retraining is expensive and impractical in a streaming environment. AFU solves this with a closed-loop pipeline:

```
Stream → Monitor → Detect Violation → Attribute Cause → Selective Unlearn → Re-evaluate → Audit
```

Key properties:
- **No full retraining** — Newton-step and gradient-reversal unlearning are orders of magnitude faster
- **Rollback safety** — every unlearning action saves parameters; if utility drops, the model reverts
- **Adaptive thresholds** — the audit logger tightens or loosens the fairness threshold based on violation frequency
- **Pure NumPy** — no PyTorch or JAX dependency; influence functions are exact for convex models

---

## Architecture

```
adaptive_fairness_unlearning/
├── utils/
│   ├── types.py            # Shared dataclasses: DataBatch, FairnessSnapshot, etc.
│   └── helpers.py          # Pure functions: sigmoid, positive-rate helpers
├── data/
│   └── stream.py           # SyntheticStreamGenerator + RealWorldStreamLoader
├── models/
│   └── base_model.py       # OnlineLogisticRegression (SGD, gradient, HVP)
├── monitors/
│   └── fairness_monitor.py # Sliding-window SPD and EOD computation
├── detectors/
│   └── bias_detector.py    # CUSUM consecutive-violation trigger
├── attribution/
│   └── influence.py        # LiSSA Hessian-inverse + fairness gradient scoring
├── unlearning/
│   └── engine.py           # Three mechanisms: Newton / gradient-reversal / reweight
├── utility/
│   └── preservation.py     # ReplayBuffer, EWC regularisation, accuracy recovery
├── audit/
│   └── logger.py           # Audit trail, summary stats, adaptive threshold
├── evaluation/
│   └── benchmarks.py       # Three baselines + Evaluator
└── pipeline.py             # Algorithm 1 — the main streaming loop
run_demo.py                 # Standalone full demonstration
```

---

## Fairness Metrics

| Metric | Formula |
|--------|---------|
| Statistical Parity Difference (SPD) | \|P(ŷ=1\|A=0) − P(ŷ=1\|A=1)\| |
| Equalized Odds Difference (EOD) | 0.5 · (\|TPR₀−TPR₁\| + \|FPR₀−FPR₁\|) |

A violation is declared when either metric exceeds threshold τ for k consecutive monitoring windows.

---

## Installation

**Requirements:** Python 3.8+, NumPy, SciPy, scikit-learn, matplotlib (optional, for plots)

```bash
git clone https://github.com/<your-org>/adaptive_fairness_unlearning.git
cd adaptive_fairness_unlearning
pip install numpy scipy scikit-learn matplotlib
```

No `setup.py` is needed for local use — run scripts from the repo root where the `adaptive_fairness_unlearning/` package directory lives.

---

## Quick Start

### Run the full demo

```bash
python run_demo.py
```

Expected output:

```
Method                 |    SPD |    EOD |    Acc | Violations |  Events |  Time(s)
-----------------------------------------------------------------------------------
afu                    | 0.0292 | 0.0561 | 0.6391 |          5 |       1 |     0.02
periodic               | 0.0276 | 0.0603 | 0.6648 |          8 |       0 |     0.02
fairness_sgd           | 0.0314 | 0.0625 | 0.6514 |          8 |       0 |     0.02
static                 | 0.0391 | 0.0666 | 0.6408 |          9 |       0 |     0.01
```

AFU achieves the fewest violations with comparable accuracy, and each unlearning event takes ~2 ms vs. a full retrain.

---

### Use the pipeline in your own code

```python
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.data.stream import SyntheticStreamGenerator
from adaptive_fairness_unlearning.pipeline import AdaptiveFairUnlearningPipeline

cfg = FrameworkConfig(
    n_batches=50,
    batch_size=200,
    fairness_threshold=0.10,
    consecutive_violations=3,
    unlearning_budget=30,
)

gen = SyntheticStreamGenerator(cfg)
stream = gen.stream(
    drift_schedule={0: 0.0, 20: 0.0, 35: 0.785},   # 45° rotation at batch 35
    bias_injection_windows=[(25, 40, 0.8)],          # strong bias injected
)

pipe = AdaptiveFairUnlearningPipeline(cfg)
history, actions = pipe.run(stream)

print(f"Processed {len(history)} batches, {len(actions)} unlearning events")
for action in actions:
    print(f"  t={action.timestamp} method={action.method} "
          f"SPD {action.fairness_before.spd:.4f} → {action.fairness_after.spd:.4f} "
          f"accepted={action.accepted}")
```

### Use a real-world dataset

```python
import numpy as np
from adaptive_fairness_unlearning.data.stream import RealWorldStreamLoader
from adaptive_fairness_unlearning.pipeline import AdaptiveFairUnlearningPipeline
from adaptive_fairness_unlearning.utils import FrameworkConfig

# X, y, protected are numpy arrays; time_order gives the temporal sort key
loader = RealWorldStreamLoader(X, y, protected, time_order)

cfg = FrameworkConfig(n_features=X.shape[1])
pipe = AdaptiveFairUnlearningPipeline(cfg)
history, actions = pipe.run(loader.stream(batch_size=500))
```

---

## Configuration

All hyperparameters live in `FrameworkConfig` with sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_features` | 10 | Feature dimensionality |
| `batch_size` | 200 | Stream batch size |
| `n_batches` | 50 | Number of batches to generate |
| `fairness_window_size` | 1000 | Sliding window length for SPD/EOD |
| `fairness_threshold` | 0.10 | Violation threshold τ |
| `consecutive_violations` | 3 | k consecutive windows before triggering |
| `influence_sample_size` | 200 | Mini-batch size for LiSSA Hessian approximation |
| `top_k_influential` | 20 | Points scored per attribution pass |
| `unlearning_budget` | 30 | Max candidates per unlearning event |
| `replay_buffer_size` | 500 | Samples retained for utility preservation |
| `accuracy_drop_tolerance` | 0.05 | Max allowed accuracy drop (5%) before recovery |
| `ewc_lambda` | 0.5 | Elastic Weight Consolidation penalty weight |
| `learning_rate` | 0.01 | SGD learning rate |
| `seed` | 42 | Global random seed |

---

## Unlearning Mechanisms

The engine auto-selects the method based on how many candidates are identified:

| Condition | Method | How it works |
|-----------|--------|-------------|
| \|U\| ≤ 20 | `influence_newton` | Newton step: θ_new = θ + (1/N) H⁻¹ Σ ∇L(z) — fast and principled |
| 20 < \|U\| ≤ 100 | `gradient_reversal` | Gradient ascent on harmful points + descent on replay buffer |
| \|U\| > 100 | `reweight` | Zero out harmful point weights; re-optimise on replay buffer |

---

## Evaluation Baselines

| Baseline | Description |
|----------|-------------|
| `PeriodicRetrainer` | Full retrain from scratch every 500 samples |
| `FairnessRegularizedSGD` | Online SGD with fairness penalty λ(SPD² + EOD²) at each step |
| `StaticUnlearner` | One-shot unlearning at deployment; plain SGD thereafter |

Run all four methods on the same stream with:

```python
from adaptive_fairness_unlearning.evaluation.benchmarks import Evaluator
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.data.stream import SyntheticStreamGenerator

cfg = FrameworkConfig(n_batches=50, batch_size=200)

def make_stream():
    return SyntheticStreamGenerator(cfg).stream(bias_injection_windows=[(25, 40, 0.8)])

results = Evaluator(cfg).run_all(make_stream)
for name, r in results.items():
    print(f"{name:20s} SPD={r['spd_mean']:.4f} Violations={r['n_violations']}")
```

---

## Module-by-Module Reference

### `data/stream.py`

**`SyntheticStreamGenerator(cfg)`**
- `.stream(drift_schedule, bias_injection_windows)` → generator of `DataBatch`
- `drift_schedule`: `Dict[batch_idx, rotation_angle_radians]` — linearly interpolated between keyframes
- `bias_injection_windows`: `List[(start, end, strength)]` — increases protected attribute's causal influence on labels during the specified batch range

**`RealWorldStreamLoader(X, y, protected, time_order)`**
- `.stream(batch_size)` → generator of `DataBatch`

---

### `models/base_model.py`

**`OnlineLogisticRegression(n_features, lr)`**

| Method | Description |
|--------|-------------|
| `predict(X)` | Returns P(y=1\|x) probabilities |
| `predict_labels(X)` | Binary predictions (threshold 0.5) |
| `update(X, y)` | Mini-batch SGD step |
| `get_params() / set_params(θ)` | Parameter snapshot and restore |
| `gradient(X, y)` | Per-sample gradients, shape (n, d+1) |
| `hessian_vector_product(X, v)` | H·v approximation for LiSSA |

---

### `monitors/fairness_monitor.py`

**`FairnessMonitor(window_size)`**
- `.update(y_pred, y_true, protected)` — push a batch, evict oldest entries
- `.spd()`, `.eod()` — current metric values on the window
- `.snapshot(timestamp, accuracy)` → `FairnessSnapshot`
- `.reset()` — clear the window (called post-unlearning)

---

### `attribution/influence.py`

**`InfluenceEstimator(model, cfg)`**

Influence score for training point z on the fairness loss:

```
I_F(z) ≈ −v^T · ∇θ L(z, θ)    where v = H⁻¹ · ∇θ L_fair
```

The Hessian inverse is approximated via **LiSSA** (Linear time Stochastic Second-order Algorithm).

- `.compute_influences(X_train, y_train, prot_train, X_eval, y_eval, prot_eval)` → scores array of shape (n_train,)
- `.get_top_k(scores, k, batch_timestamps)` → `List[UnlearningCandidate]`

---

### `audit/logger.py`

**`AuditLogger()`**

Every unlearning event records: timestamp, candidates, fairness before/after, utility before/after, method, wall-clock cost, and accept/rollback decision.

- `.summary_stats()` — aggregate statistics dict
- `.adaptive_threshold(current_tau)` — tighten τ by 10% if >50% of recent events were violations; loosen by 5% if <20%; clamped to [0.03, 0.20]
- `.to_dataframe()` — dict-of-lists compatible with `pd.DataFrame`
- `.explainability_report(action)` — top-k influential points and fairness delta for a given event

---

## Design Decisions

**Convex-first.** The initial implementation uses logistic regression — influence functions are exact, Hessians are tractable, and unlearning is provably correct. To extend to neural networks, implement a model class with `gradient()` and `hessian_vector_product()` methods.

**Hierarchical attribution.** Full point-wise influence on the entire stream is prohibitively expensive. The pipeline screens at the replay-buffer level first (cheap), then ranks points within flagged batches only.

**Rollback safety.** Every unlearning action saves `old_params`. If utility drops beyond tolerance after EWC recovery attempts, the model rolls back and the event is logged as rejected — no silent degradation.

**Adaptive thresholds.** The audit logger's feedback loop prevents both alarm fatigue (threshold too tight, too many triggers) and under-detection (threshold too loose, violations ignored).

---

## Dependencies

| Library | Use |
|---------|-----|
| NumPy | Core array operations, linear algebra |
| SciPy | Optional conjugate gradient solver |
| scikit-learn | AUC computation in baselines |
| matplotlib | Fairness trajectory plots (optional) |

No PyTorch, JAX, or TensorFlow required.

---

## Research Context

This framework implements the methodology from:

- **Plan 1**: *Step-by-Step Implementation Plan: Adaptive Unlearning for Dynamic Fairness* — 10-step methodology covering CUSUM trigger, Newton-step unlearning, LiSSA Hessian approximation, and EWC regularisation.
- **Plan 2**: *Step-by-Step Implementation Plan for Adaptive Unlearning for Dynamic Fairness in Changing Data Distributions* — formal constrained optimisation framing, hierarchical attribution (batch → point → feature), three unlearning mechanisms, and rollback logic.

The formal optimisation objective is:

```
min  L_task(θ) + λ_u · L_unlearn(θ) + λ_s · L_stability(θ)
 θ
subject to  F(θ; W_t) ≤ δ_f
```

where `F(θ; W_t)` is the fairness violation measured on the current sliding window `W_t`, and `δ_f` is the acceptable fairness threshold.
