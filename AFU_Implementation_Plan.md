# Adaptive Fairness Unlearning — Implementation Plan

## Project Overview

This document lays out the full project structure, module contracts, data flow, and build order for the **Adaptive Fairness Unlearning (AFU)** framework. The framework continuously monitors a deployed model's fairness, detects bias re-emergence from distribution shifts or biased injections, and triggers selective unlearning of problematic data points — all without full retraining.

The design synthesises both implementation plans (Plan 1's 10-step methodology and Plan 2's six-module architecture with formal optimisation framing).

---

## Directory Structure

```
adaptive_fairness_unlearning/
│
├── utils/
│   ├── __init__.py
│   ├── types.py            # Shared data structures & config
│   └── helpers.py           # Pure utility functions
│
├── data/
│   ├── __init__.py
│   └── stream.py            # Step 1 — Data stream simulation & preprocessing
│
├── models/
│   ├── __init__.py
│   └── base_model.py        # Step 2 — Base model deployment
│
├── monitors/
│   ├── __init__.py
│   └── fairness_monitor.py  # Step 3 — Online fairness monitoring
│
├── detectors/
│   ├── __init__.py
│   └── bias_detector.py     # Step 4 — Bias detection trigger (CUSUM)
│
├── attribution/
│   ├── __init__.py
│   └── influence.py         # Step 5 — Bias source identification
│
├── unlearning/
│   ├── __init__.py
│   └── engine.py            # Step 6 — Selective unlearning engine
│
├── utility/
│   ├── __init__.py
│   └── preservation.py      # Step 7 — Utility preservation (EWC + replay)
│
├── audit/
│   ├── __init__.py
│   └── logger.py            # Step 8 — Audit trail & adaptive control
│
├── evaluation/
│   ├── __init__.py
│   └── benchmarks.py        # Step 9 — Baselines & evaluation metrics
│
├── pipeline.py              # Step 10 — Main loop (Algorithm 1)
└── __init__.py
```

---

## Module Specifications

### 1. `utils/types.py` — Shared Data Structures

Central type definitions imported by every other module.

| Type | Key Fields | Purpose |
|------|-----------|---------|
| `DataBatch` | `X`, `y`, `protected`, `timestamp`, `indices`, `metadata` | One chunk from the data stream |
| `FairnessSnapshot` | `timestamp`, `spd`, `eod`, `accuracy`, `window_size` | Fairness + utility at a point in time |
| `UnlearningCandidate` | `index`, `influence_score`, `batch_timestamp`, `features` | A data point flagged as a bias source |
| `UnlearningAction` | `timestamp`, `candidates`, `fairness_before/after`, `utility_before/after`, `method`, `cost_seconds`, `accepted` | Full record of one unlearning event |
| `FrameworkConfig` | All hyperparameters (window sizes, thresholds, budgets, learning rate, etc.) | Single source of truth for configuration |

### 2. `utils/helpers.py` — Pure Utility Functions

No state, no side effects.

| Function | Signature | Description |
|----------|-----------|-------------|
| `sigmoid` | `(x: ndarray) → ndarray` | Numerically stable sigmoid |
| `safe_positive_rate` | `(preds: ndarray) → float` | P(ŷ=1); returns 0 if array is empty |
| `conditional_positive_rate` | `(preds, labels, groups, group, label) → float` | P(ŷ=1 \| Y=label, A=group) |

---

### 3. `data/stream.py` — Step 1: Data Stream Simulation & Preprocessing

Two classes, both yielding `DataBatch` objects through a `.stream()` generator so the pipeline consumes them identically.

**`SyntheticStreamGenerator(cfg: FrameworkConfig)`**

- Generates binary-classification streaming data with N(0, I) features.
- **Concept drift**: decision boundary rotates over time via a `drift_schedule: Dict[int, float]` mapping batch index → rotation angle (linearly interpolated between keyframes).
- **Bias injection**: `bias_injection_windows: List[Tuple[start, end, strength]]` — during `[start, end)`, the protected attribute's causal influence on the label increases by `strength`.
- Method: `.stream(drift_schedule, bias_injection_windows) → Generator[DataBatch]`

**`RealWorldStreamLoader(X, y, protected, time_order)`**

- Wraps any pre-loaded tabular dataset (e.g., Home Credit Default, MIMIC-IV).
- Sorts by `time_order`, chunks into sequential batches.
- Method: `.stream(batch_size) → Generator[DataBatch]`

**Protected attributes**: binary {0, 1}. Fairness metrics defined downstream.

---

### 4. `models/base_model.py` — Step 2: Base Model Deployment

**`OnlineLogisticRegression(n_features, lr)`**

Stores parameters `self.w` (weight vector) and `self.b` (bias scalar) as numpy arrays.

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict` | `(X) → ndarray` | Returns probabilities via sigmoid(Xw + b) |
| `predict_labels` | `(X) → ndarray` | Binary predictions (threshold 0.5) |
| `update` | `(X, y)` | One mini-batch SGD step |
| `get_params` | `() → ndarray` | Returns a copy of θ = [w; b] |
| `set_params` | `(θ)` | Restores parameters (used for rollback) |
| `gradient` | `(X, y) → ndarray` | Per-sample gradient ∇θ L(z, θ) |
| `hessian_vector_product` | `(X, v) → ndarray` | H·v approximation (for influence functions) |

---

### 5. `monitors/fairness_monitor.py` — Step 3: Online Fairness Monitoring

**`FairnessMonitor(window_size: int)`**

Maintains a fixed-size sliding window of `(ŷ, y, A)` triples.

| Method | Signature | Description |
|--------|-----------|-------------|
| `update` | `(y_pred, y_true, protected)` | Push a batch into the window; evict oldest if overflow |
| `spd` | `() → float` | SPD = \|P(ŷ=1\|A=0) − P(ŷ=1\|A=1)\| |
| `eod` | `() → float` | EOD = 0.5 · (\|TPR₀ − TPR₁\| + \|FPR₀ − FPR₁\|) |
| `snapshot` | `(timestamp) → FairnessSnapshot` | Bundles current metrics into a snapshot |
| `reset` | `()` | Clears the window |

---

### 6. `detectors/bias_detector.py` — Step 4: Bias Detection Trigger

**`BiasDetector(threshold: float, consecutive_k: int)`**

Implements a CUSUM (cumulative sum) control chart over the fairness metrics stream.

| Method | Signature | Description |
|--------|-----------|-------------|
| `check` | `(snapshot: FairnessSnapshot) → bool` | Returns `True` when SPD or EOD exceeds τ for k consecutive checks |
| `reset` | `()` | Resets internal CUSUM state (post-unlearning cooldown) |
| `get_cumsum` | `() → dict` | Exposes internal state for debugging/audit |

**Trigger logic**: a fairness violation flag is raised only when *both* conditions hold — (a) the metric exceeds the threshold, and (b) it has persisted for k consecutive windows. This two-level check avoids false alarms from random noise (per Plan 2, Step 4).

---

### 7. `attribution/influence.py` — Step 5: Bias Source Identification

**`InfluenceEstimator(model, cfg: FrameworkConfig)`**

Identifies which training points contribute most to the current bias using influence functions.

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_influences` | `(X_train, y_train, prot_train, X_eval, y_eval, prot_eval) → ndarray` | Influence score for each training point on the fairness loss |
| `get_top_k` | `(scores, k) → List[UnlearningCandidate]` | Returns the k most harmful points |

**Core formula**:

```
I_F(z) ≈ −∇θ L_fair(θ)ᵀ · H⁻¹ · ∇θ L(z, θ)
```

Where:
- `L_fair` is a fairness loss (e.g., TPR difference between groups)
- `H` is the Hessian of the training loss
- Hessian inverse is approximated via **LiSSA** (stochastic approximation) on a random subset of size `influence_sample_size`

**Hierarchical attribution** (Plan 2, Step 5): batch-level screening first → point-level influence ranking on candidate batches only → optional feature-level SHAP attribution for proxy detection.

---

### 8. `unlearning/engine.py` — Step 6: Selective Unlearning Engine

**`SelectiveUnlearner(model, cfg: FrameworkConfig)`**

Three mechanisms exposed through a single dispatcher.

| Method | Signature | Description |
|--------|-----------|-------------|
| `unlearn` | `(candidates, method, replay_buffer) → ndarray` | Dispatches to the chosen mechanism; returns updated params |

**Mechanism A — `"influence_newton"`** (primary):
```
θ_new = θ_old + (1/N) · H⁻¹ · Σ_{z ∈ U} ∇θ L(z, θ_old)
```
Fast, principled, good for small candidate sets.

**Mechanism B — `"gradient_reversal"`**:
A few SGD steps *ascending* on the harmful points' loss, with a simultaneous *descending* step on the replay buffer to preserve utility.

**Mechanism C — `"reweight"`**:
Sets harmful point weights to zero, re-optimises the loss on the replay buffer with adjusted weights.

**Dispatch rule**: use Newton for |U| ≤ 20, gradient reversal for 20 < |U| ≤ 100, reweight/SISA fallback for |U| > 100.

---

### 9. `utility/preservation.py` — Step 7: Utility Preservation

**`ReplayBuffer(max_size: int)`**

Maintains a reservoir of recent, fairly-behaved data points.

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(batch: DataBatch)` | Reservoir sampling into the buffer |
| `sample` | `(n) → DataBatch` | Random sample for fine-tuning |
| `get_all` | `() → DataBatch` | Full buffer contents |

**`UtilityPreserver(model, cfg: FrameworkConfig)`**

| Method | Signature | Description |
|--------|-----------|-------------|
| `fine_tune` | `(model, buffer, steps)` | Gradient steps on replay memory |
| `ewc_regularize` | `(model, old_params, fisher)` | Elastic weight consolidation penalty |
| `check_and_recover` | `(model, val_batch, old_accuracy) → bool` | If accuracy dropped > δ, triggers recovery; returns success/fail |

**Recovery logic**: after unlearning, evaluate accuracy on a validation window. If drop > δ (default 5%), apply EWC-regularised fine-tuning on the replay buffer. If still failing, signal rollback.

---

### 10. `audit/logger.py` — Step 8: Audit Trail & Adaptive Control

**`AuditLogger()`**

| Method | Signature | Description |
|--------|-----------|-------------|
| `log` | `(action: UnlearningAction)` | Appends an event record |
| `get_history` | `() → List[UnlearningAction]` | Full audit trail |
| `summary_stats` | `() → dict` | Total events, mean cost, violation frequency, acceptance rate |
| `adaptive_threshold` | `(current_tau) → float` | Feedback loop: if violations recur frequently, tighten τ |
| `explainability_report` | `(action) → dict` | Top-k features/points causing bias for a given event |

**Logged per event**: timestamp, violated metric(s), number of removed points, influence scores, fairness before/after, utility before/after, computational cost, accept/rollback decision.

---

### 11. `evaluation/benchmarks.py` — Step 9: Evaluation Protocol

**`Evaluator(cfg: FrameworkConfig)`**

Runs the AFU pipeline and three baselines on the same stream, collecting comparable metrics.

**Baselines**:

| Baseline | Class | Description |
|----------|-------|-------------|
| Periodic full retraining | `PeriodicRetrainer` | Retrain from scratch every N samples |
| Fairness-regularised SGD | `FairnessRegularizedSGD` | Online SGD with a fairness penalty term at each step |
| Static unlearning | `StaticUnlearner` | One-shot unlearning at deployment, no adaptation |

**Evaluation metrics** (four dimensions):

| Dimension | Metrics |
|-----------|---------|
| Fairness stability | Mean/std/max of SPD and EOD over time; number of violations; recovery time |
| Predictive utility | AUC, accuracy, group-wise performance on hold-out stream |
| Computational cost | Wall-clock time, memory overhead |
| Unlearning efficiency | Time per unlearning event vs. full retraining |

Method: `.run_all(stream) → dict` returns a structured results dictionary.

---

### 12. `pipeline.py` — Step 10: Main Loop (Algorithm 1)

**`AdaptiveFairUnlearningPipeline(cfg: FrameworkConfig)`**

Wires all modules together into the streaming main loop.

```
Initialise: model, monitor, detector, influence_estimator,
            unlearner, preserver, replay_buffer, audit_logger

for each batch Bt from stream:
    ŷ ← model.predict(Bt)
    monitor.update(ŷ, Bt.y, Bt.protected)
    snapshot ← monitor.snapshot(t)

    if detector.check(snapshot):                      # FAIRNESS VIOLATION
        scores ← influence.compute_influences(...)
        candidates ← influence.get_top_k(scores, budget)
        old_params ← model.get_params()
        old_acc ← snapshot.accuracy

        unlearner.unlearn(candidates, method)

        new_snapshot ← re-evaluate on window
        if preserver.check_and_recover(model, val, old_acc):
            audit.log(accepted action)
            detector.reset()                          # cooldown
        else:
            model.set_params(old_params)              # ROLLBACK
            audit.log(rejected action)
    else:
        model.update(Bt.X, Bt.y)                      # standard online learning

    replay_buffer.add(Bt)
```

Method: `.run(stream) → (history: List[FairnessSnapshot], audit: List[UnlearningAction])`

---

## Data Flow Summary

```
Stream ──→ Model.predict ──→ FairnessMonitor ──→ BiasDetector
                                                      │
                                          [no violation] │ [violation]
                                                │        │
                                      Model.update    InfluenceEstimator
                                      (standard SGD)      │
                                                   SelectiveUnlearner
                                                      │
                                                UtilityPreserver
                                                   │         │
                                              [accept]   [rollback]
                                                   │         │
                                              AuditLogger ←──┘
```

---

## Build Order

Each layer depends only on what came before.

| Phase | Module | Dependencies |
|-------|--------|-------------|
| 1 | `utils/types.py`, `utils/helpers.py` | numpy only |
| 2 | `data/stream.py` | utils |
| 3 | `models/base_model.py` | utils |
| 4 | `monitors/fairness_monitor.py` | utils |
| 5 | `detectors/bias_detector.py` | utils (FairnessSnapshot) |
| 6 | `attribution/influence.py` | utils, models |
| 7 | `unlearning/engine.py` | utils, models |
| 8 | `utility/preservation.py` | utils, models |
| 9 | `audit/logger.py` | utils |
| 10 | `pipeline.py` | all of the above |
| 11 | `evaluation/benchmarks.py` | pipeline + baselines |

---

## Key Design Decisions

**Convex-first, then neural.** The initial implementation uses logistic regression — influence functions are exact, Hessians are tractable, and unlearning is provably correct. Once validated, swap in `models/mlp.py` for a non-convex model using Hessian-vector product approximations.

**Hierarchical attribution.** Full point-wise influence on the entire stream is too expensive. We screen at the batch level first (cheap), then rank points within flagged batches only (Plan 2, Step 5).

**Rollback safety.** Every unlearning action saves old parameters. If utility drops beyond tolerance after recovery attempts, we rollback and log the failure. No silent degradation.

**Adaptive thresholds.** The audit logger's feedback loop tightens or loosens the fairness threshold τ based on violation frequency — preventing both alarm fatigue and under-detection.

---

## Libraries

| Library | Use |
|---------|-----|
| NumPy | Core array operations, linear algebra |
| SciPy | Hessian approximation (conjugate gradient), statistical tests (KS) |
| scikit-learn | Baseline models, AUC computation |
| matplotlib | Fairness trajectory plots, evaluation charts |

No PyTorch/JAX dependency in the initial implementation — pure NumPy for transparency and reproducibility.
