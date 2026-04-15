# AFU Framework — Claude Code Run Plan

## Context

You are building the **Adaptive Fairness Unlearning (AFU)** framework — a Python library that monitors a deployed model's fairness in a streaming setting, detects bias re-emergence, identifies harmful data points via influence functions, selectively unlearns them without full retraining, and preserves model utility.

There are two reference PDFs in the project directory:
- `RP3Implementation_Plan1_1.pdf` — 10-step methodology with specific algorithms (Newton-step unlearning, CUSUM trigger, LiSSA Hessian approx, EWC regularisation)
- `RP3Implementation_Plan2.pdf` — formal architecture with constrained optimisation framing, hierarchical attribution, three unlearning mechanisms, rollback logic, and evaluation design

Read both documents before starting. The implementation plan below synthesises them.

---

## Phase 1: Scaffold and Shared Types

Create the full directory structure:

```
adaptive_fairness_unlearning/
├── utils/
│   ├── __init__.py
│   ├── types.py
│   └── helpers.py
├── data/
│   ├── __init__.py
│   └── stream.py
├── models/
│   ├── __init__.py
│   └── base_model.py
├── monitors/
│   ├── __init__.py
│   └── fairness_monitor.py
├── detectors/
│   ├── __init__.py
│   └── bias_detector.py
├── attribution/
│   ├── __init__.py
│   └── influence.py
├── unlearning/
│   ├── __init__.py
│   └── engine.py
├── utility/
│   ├── __init__.py
│   └── preservation.py
├── audit/
│   ├── __init__.py
│   └── logger.py
├── evaluation/
│   ├── __init__.py
│   └── benchmarks.py
├── pipeline.py
├── __init__.py
└── run_demo.py
```

**`utils/types.py`** — Define these dataclasses:

- `DataBatch`: fields `X: np.ndarray`, `y: np.ndarray`, `protected: np.ndarray`, `timestamp: int`, `indices: Optional[np.ndarray]`, `metadata: dict`
- `FairnessSnapshot`: fields `timestamp: int`, `spd: float`, `eod: float`, `accuracy: float`, `window_size: int`, `extra: dict`
- `UnlearningCandidate`: fields `index: int`, `influence_score: float`, `batch_timestamp: int`, `features: Optional[np.ndarray]`
- `UnlearningAction`: fields `timestamp: int`, `candidates: List[UnlearningCandidate]`, `fairness_before: FairnessSnapshot`, `fairness_after: FairnessSnapshot`, `utility_before: float`, `utility_after: float`, `method: str`, `cost_seconds: float`, `accepted: bool`, `notes: str`
- `FrameworkConfig`: all hyperparameters with sensible defaults:
  - `n_features=10`, `n_informative=5`, `batch_size=200`, `n_batches=50`, `noise_std=0.3`, `base_protected_correlation=0.05`, `seed=42`
  - `fairness_window_size=1000`, `fairness_threshold=0.10`, `consecutive_violations=3`
  - `influence_sample_size=200`, `top_k_influential=20`
  - `unlearning_budget=30`, `replay_buffer_size=500`
  - `accuracy_drop_tolerance=0.05`, `ewc_lambda=0.5`
  - `learning_rate=0.01`

**`utils/helpers.py`** — Pure functions: `sigmoid(x)`, `safe_positive_rate(preds)`, `conditional_positive_rate(preds, labels, groups, group, label)`.

After creating these, run a quick import test:
```bash
python -c "from adaptive_fairness_unlearning.utils import DataBatch, FrameworkConfig, sigmoid; print('Phase 1 OK')"
```

---

## Phase 2: Data Stream Module

**`data/stream.py`** — Two classes:

### `SyntheticStreamGenerator(cfg: FrameworkConfig)`

- In `__init__`: seed an `np.random.RandomState`, generate a random unit-normed base weight vector of size `n_informative`, and an orthogonal drift direction vector.
- `.stream(drift_schedule: Dict[int, float] = None, bias_injection_windows: List[Tuple[int, int, float]] = None) -> Generator[DataBatch]`:
  - For each batch t in range(n_batches):
    - Draw X ~ N(0, I) of shape (batch_size, n_features)
    - Draw protected ~ Bernoulli(0.5) of shape (batch_size,)
    - Compute rotated weights: w = cos(angle)*base + sin(angle)*drift_dir, where angle is linearly interpolated from drift_schedule
    - Compute logit = X[:, :n_informative] @ w + base_correlation * (2*protected - 1) + bias_injection * (2*protected - 1) + noise
    - Label via y = Bernoulli(sigmoid(logit))
    - Yield DataBatch with global indices tracking

### `RealWorldStreamLoader(X, y, protected, time_order)`

- Sort all arrays by time_order in __init__
- `.stream(batch_size=500) -> Generator[DataBatch]`: chunk sequentially

**Test**:
```bash
python -c "
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.data.stream import SyntheticStreamGenerator
cfg = FrameworkConfig(n_batches=5)
gen = SyntheticStreamGenerator(cfg)
for batch in gen.stream():
    print(f't={batch.timestamp} X={batch.X.shape} y_mean={batch.y.mean():.2f}')
print('Phase 2 OK')
"
```

---

## Phase 3: Base Model

**`models/base_model.py`** — `OnlineLogisticRegression(n_features: int, lr: float)`

Store `self.w = np.zeros(n_features)` and `self.b = 0.0`.

Methods:
- `predict(X) -> ndarray`: returns sigmoid(X @ w + b)
- `predict_labels(X) -> ndarray`: returns (predict(X) >= 0.5).astype(int)
- `update(X, y)`: one mini-batch SGD step. grad_logits = predict(X) - y; w -= lr * (X.T @ grad_logits) / len(y); b -= lr * grad_logits.mean()
- `get_params() -> ndarray`: returns np.concatenate([w, [b]]).copy()
- `set_params(theta)`: w = theta[:-1]; b = theta[-1]
- `gradient(X, y) -> ndarray`: returns per-sample gradients, shape (n_samples, n_features+1). For each sample: grad_w = (sigmoid(x@w+b) - y) * x, grad_b = sigmoid(x@w+b) - y
- `hessian_vector_product(X, v) -> ndarray`: computes H·v where H is the Hessian of the logistic loss. H = (1/N) X_aug.T @ diag(p*(1-p)) @ X_aug where X_aug includes the bias column and p = sigmoid predictions. Return H @ v.

**Test**:
```bash
python -c "
import numpy as np
from adaptive_fairness_unlearning.models.base_model import OnlineLogisticRegression
model = OnlineLogisticRegression(10, 0.01)
X = np.random.randn(50, 10)
y = np.random.randint(0, 2, 50)
model.update(X, y)
print(f'preds shape: {model.predict(X).shape}')
print(f'params shape: {model.get_params().shape}')
print('Phase 3 OK')
"
```

---

## Phase 4: Fairness Monitor

**`monitors/fairness_monitor.py`** — `FairnessMonitor(window_size: int)`

Internally stores three Python lists: `_preds`, `_labels`, `_groups`. On `.update(y_pred, y_true, protected)`, extend all three and trim to window_size.

- `.spd() -> float`: |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
- `.eod() -> float`: 0.5 * (|TPR_0 - TPR_1| + |FPR_0 - FPR_1|)
- `.snapshot(timestamp, accuracy) -> FairnessSnapshot`
- `.reset()`: clears all lists

**Test**:
```bash
python -c "
import numpy as np
from adaptive_fairness_unlearning.monitors.fairness_monitor import FairnessMonitor
fm = FairnessMonitor(500)
fm.update(np.array([1,1,0,0,1]), np.array([1,0,0,1,1]), np.array([0,0,0,1,1]))
print(f'SPD={fm.spd():.4f} EOD={fm.eod():.4f}')
print('Phase 4 OK')
"
```

---

## Phase 5: Bias Detector

**`detectors/bias_detector.py`** — `BiasDetector(threshold: float, consecutive_k: int)`

Track `_violation_count` (int). On `.check(snapshot: FairnessSnapshot) -> bool`:
- If snapshot.spd > threshold or snapshot.eod > threshold: increment _violation_count
- Else: reset _violation_count to 0
- Return _violation_count >= consecutive_k

Also: `.reset()` zeros the counter, `.get_state() -> dict` for debugging.

**Test**:
```bash
python -c "
from adaptive_fairness_unlearning.utils import FairnessSnapshot
from adaptive_fairness_unlearning.detectors.bias_detector import BiasDetector
bd = BiasDetector(0.10, 3)
for i in range(5):
    snap = FairnessSnapshot(i, spd=0.15, eod=0.12, accuracy=0.8, window_size=500)
    triggered = bd.check(snap)
    print(f't={i} triggered={triggered}')
print('Phase 5 OK')
"
```

Expected: triggered=False for t=0,1 and triggered=True for t=2,3,4.

---

## Phase 6: Influence-Based Attribution

**`attribution/influence.py`** — `InfluenceEstimator(model, cfg: FrameworkConfig)`

### `compute_influences(X_train, y_train, prot_train, X_eval, y_eval, prot_eval) -> ndarray`

Steps:
1. Compute the fairness gradient: ∇θ L_fair. Define L_fair as the squared difference of group-conditional positive rates. Approximate ∇θ L_fair numerically or analytically.
   - Practical approach: compute gradient of loss on group-0 eval points minus gradient of loss on group-1 eval points. This gives the direction in parameter space that increases the inter-group gap.
2. Approximate H⁻¹ · ∇θ L_fair using LiSSA (Linear time Stochastic Second-order Algorithm):
   - Initialise v = fairness_gradient
   - For T iterations (e.g., 10): sample a mini-batch from training data, compute Hessian-vector product H·v, update v = fairness_gradient + (I - H)·v
   - This converges to H⁻¹ · fairness_gradient
3. For each training point z_i, compute influence: I(z_i) = -v^T · ∇θ L(z_i, θ)
4. Return the scores array.

### `get_top_k(scores, k, batch_timestamps) -> List[UnlearningCandidate]`

Return the k points with the highest positive influence scores (most harmful to fairness).

**Test**:
```bash
python -c "
import numpy as np
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.models.base_model import OnlineLogisticRegression
from adaptive_fairness_unlearning.attribution.influence import InfluenceEstimator
cfg = FrameworkConfig()
model = OnlineLogisticRegression(10, 0.01)
X = np.random.randn(100, 10); y = np.random.randint(0,2,100); p = np.random.randint(0,2,100)
model.update(X, y)
ie = InfluenceEstimator(model, cfg)
scores = ie.compute_influences(X, y, p, X[:20], y[:20], p[:20])
print(f'scores shape: {scores.shape}, max: {scores.max():.4f}')
candidates = ie.get_top_k(scores, 5, np.zeros(100, dtype=int))
print(f'top 5 candidates: {[c.index for c in candidates]}')
print('Phase 6 OK')
"
```

---

## Phase 7: Selective Unlearning Engine

**`unlearning/engine.py`** — `SelectiveUnlearner(model, cfg: FrameworkConfig)`

### `.unlearn(candidates: List[UnlearningCandidate], X_train, y_train, method: str, replay_X=None, replay_y=None) -> ndarray`

Dispatches based on `method`:

**`"influence_newton"`**:
```python
theta = model.get_params()
grads = model.gradient(X_candidates, y_candidates)  # (|U|, d+1)
hvp = sum of H^{-1} @ grad for each candidate (use LiSSA or direct solve)
theta_new = theta + (1/N_total) * hvp
model.set_params(theta_new)
return theta_new
```

**`"gradient_reversal"`**:
```python
# Ascend on harmful points (forget them)
for step in range(5):
    grad_forget = model.gradient(X_candidates, y_candidates).mean(axis=0)
    theta = model.get_params()
    theta += lr * grad_forget  # gradient ASCENT
    # Descend on replay buffer (preserve utility)
    if replay_X is not None:
        grad_retain = model.gradient(replay_X, replay_y).mean(axis=0)
        theta -= lr * grad_retain
    model.set_params(theta)
```

**`"reweight"`**:
Zero out harmful points and re-optimise on replay buffer for a few steps.

Also include an `auto_select_method(n_candidates) -> str` that picks the method based on candidate set size (newton ≤ 20, reversal ≤ 100, reweight > 100).

**Test**:
```bash
python -c "
import numpy as np
from adaptive_fairness_unlearning.utils import FrameworkConfig, UnlearningCandidate
from adaptive_fairness_unlearning.models.base_model import OnlineLogisticRegression
from adaptive_fairness_unlearning.unlearning.engine import SelectiveUnlearner
cfg = FrameworkConfig()
model = OnlineLogisticRegression(10, 0.01)
X = np.random.randn(100, 10); y = np.random.randint(0,2,100)
model.update(X, y)
params_before = model.get_params().copy()
su = SelectiveUnlearner(model, cfg)
candidates = [UnlearningCandidate(i, 0.5, 0) for i in range(5)]
su.unlearn(candidates, X, y, 'gradient_reversal', X[50:], y[50:])
params_after = model.get_params()
print(f'params changed: {not np.allclose(params_before, params_after)}')
print('Phase 7 OK')
"
```

---

## Phase 8: Utility Preservation

**`utility/preservation.py`**

### `ReplayBuffer(max_size: int)`

- `.add(batch: DataBatch)`: append data; if over max_size, keep only the most recent max_size samples
- `.sample(n) -> Tuple[ndarray, ndarray, ndarray]`: random sample of (X, y, protected)
- `.get_all() -> Tuple[ndarray, ndarray, ndarray]`
- `.size -> int`

### `UtilityPreserver(model, cfg: FrameworkConfig)`

- `.fine_tune(replay_buffer, steps=10)`: run SGD steps on replay buffer samples
- `.compute_fisher(X, y) -> ndarray`: diagonal Fisher information matrix = mean of squared gradients. Used for EWC.
- `.ewc_regularize(old_params, fisher, steps=10, replay_buffer=None)`: fine-tune with L_total = L_task(replay) + ewc_lambda * sum(F_k * (θ_k - θ_old_k)^2)
- `.check_and_recover(val_X, val_y, old_accuracy, replay_buffer) -> bool`: evaluate current accuracy; if drop > tolerance, call ewc_regularize; return True if recovered, False if should rollback.

**Test**:
```bash
python -c "
import numpy as np
from adaptive_fairness_unlearning.utils import FrameworkConfig, DataBatch
from adaptive_fairness_unlearning.models.base_model import OnlineLogisticRegression
from adaptive_fairness_unlearning.utility.preservation import ReplayBuffer, UtilityPreserver
cfg = FrameworkConfig()
model = OnlineLogisticRegression(10, 0.01)
buf = ReplayBuffer(200)
batch = DataBatch(np.random.randn(50,10), np.random.randint(0,2,50), np.random.randint(0,2,50), 0)
buf.add(batch)
print(f'buffer size: {buf.size}')
up = UtilityPreserver(model, cfg)
up.fine_tune(buf, steps=3)
print('Phase 8 OK')
"
```

---

## Phase 9: Audit Logger

**`audit/logger.py`** — `AuditLogger()`

- `_history: List[UnlearningAction]`
- `.log(action: UnlearningAction)`: append
- `.get_history() -> List[UnlearningAction]`
- `.summary_stats() -> dict`: returns `{"total_events": int, "accepted": int, "rejected": int, "mean_cost_seconds": float, "mean_spd_improvement": float, "mean_eod_improvement": float, "violation_frequency": float}`
- `.adaptive_threshold(current_tau, lookback=10) -> float`: if more than 50% of the last `lookback` events were violations, tighten tau by 10%; if fewer than 20%, loosen by 5%. Clamp between 0.03 and 0.20.
- `.to_dataframe() -> dict` (or pandas DataFrame if available): tabular view of all events for analysis

**Test**:
```bash
python -c "
from adaptive_fairness_unlearning.utils import FairnessSnapshot, UnlearningAction
from adaptive_fairness_unlearning.audit.logger import AuditLogger
al = AuditLogger()
snap = FairnessSnapshot(0, 0.15, 0.12, 0.85, 500)
action = UnlearningAction(0, [], snap, snap, 0.85, 0.83, 'newton', 0.5, True)
al.log(action)
print(al.summary_stats())
print(f'adaptive tau: {al.adaptive_threshold(0.10):.3f}')
print('Phase 9 OK')
"
```

---

## Phase 10: Main Pipeline

**`pipeline.py`** — `AdaptiveFairUnlearningPipeline(cfg: FrameworkConfig)`

In `__init__`, instantiate all modules: model, fairness_monitor, bias_detector, influence_estimator, selective_unlearner, utility_preserver, replay_buffer, audit_logger.

### `.run(stream: Generator[DataBatch]) -> Tuple[List[FairnessSnapshot], List[UnlearningAction]]`

```python
history = []
for batch in stream:
    # Predict
    y_pred = model.predict_labels(batch.X)
    accuracy = (y_pred == batch.y).mean()

    # Monitor fairness
    monitor.update(y_pred, batch.y, batch.protected)
    snapshot = monitor.snapshot(batch.timestamp, accuracy)
    history.append(snapshot)

    # Check for violation
    if detector.check(snapshot):
        import time
        t0 = time.time()

        # Attribution
        buf_X, buf_y, buf_p = replay_buffer.get_all()
        scores = influence_estimator.compute_influences(
            buf_X, buf_y, buf_p, batch.X, batch.y, batch.protected
        )
        candidates = influence_estimator.get_top_k(
            scores, cfg.unlearning_budget, ...
        )

        # Save state for rollback
        old_params = model.get_params()
        old_acc = accuracy

        # Unlearn
        method = unlearner.auto_select_method(len(candidates))
        rep_X, rep_y, _ = replay_buffer.sample(min(100, replay_buffer.size))
        unlearner.unlearn(candidates, buf_X, buf_y, method, rep_X, rep_y)

        # Utility check + recovery
        val_X, val_y = batch.X, batch.y  # use current batch as validation
        recovered = preserver.check_and_recover(val_X, val_y, old_acc, replay_buffer)

        cost = time.time() - t0

        # Re-evaluate fairness
        new_preds = model.predict_labels(batch.X)
        monitor_temp = FairnessMonitor(cfg.fairness_window_size)
        monitor_temp.update(new_preds, batch.y, batch.protected)
        new_snapshot = monitor_temp.snapshot(batch.timestamp, (new_preds == batch.y).mean())

        if recovered:
            action = UnlearningAction(
                batch.timestamp, candidates, snapshot, new_snapshot,
                old_acc, (new_preds == batch.y).mean(), method, cost, True
            )
            audit.log(action)
            detector.reset()
        else:
            model.set_params(old_params)  # ROLLBACK
            action = UnlearningAction(
                batch.timestamp, candidates, snapshot, snapshot,
                old_acc, old_acc, method, cost, False, "rollback"
            )
            audit.log(action)

        # Adaptive threshold
        cfg.fairness_threshold = audit.adaptive_threshold(cfg.fairness_threshold)
    else:
        # Standard online learning
        model.update(batch.X, batch.y)

    # Update replay buffer
    replay_buffer.add(batch)

return history, audit.get_history()
```

**Test**:
```bash
python -c "
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.data.stream import SyntheticStreamGenerator
from adaptive_fairness_unlearning.pipeline import AdaptiveFairUnlearningPipeline
import numpy as np
cfg = FrameworkConfig(n_batches=10, batch_size=100)
gen = SyntheticStreamGenerator(cfg)
pipe = AdaptiveFairUnlearningPipeline(cfg)
history, actions = pipe.run(gen.stream(
    bias_injection_windows=[(4, 8, 0.8)]
))
print(f'batches processed: {len(history)}')
print(f'unlearning events: {len(actions)}')
for h in history:
    print(f'  t={h.timestamp} SPD={h.spd:.4f} EOD={h.eod:.4f} acc={h.accuracy:.4f}')
print('Phase 10 OK')
"
```

---

## Phase 11: Evaluation Benchmarks

**`evaluation/benchmarks.py`**

### Baseline classes

Each has `.run(stream) -> List[FairnessSnapshot]`:

1. **`PeriodicRetrainer(cfg, retrain_every=1000)`**: accumulates data; every `retrain_every` samples, retrains the model from scratch on all accumulated data.
2. **`FairnessRegularizedSGD(cfg, lambda_fair=0.1)`**: standard online SGD but adds a fairness penalty term: L = L_task + lambda_fair * (SPD^2 + EOD^2) at each update step.
3. **`StaticUnlearner(cfg)`**: runs one-shot unlearning on the first batch, then does plain online SGD with no further adaptation.

### `Evaluator(cfg: FrameworkConfig)`

`.run_all(stream_factory) -> dict`:
- `stream_factory` is a callable that returns a fresh stream (so each method gets the same data)
- Runs AFU pipeline + all 3 baselines
- Returns dict with keys `["afu", "periodic", "fairness_sgd", "static"]`, each containing:
  - `spd_mean`, `spd_std`, `spd_max`
  - `eod_mean`, `eod_std`, `eod_max`
  - `accuracy_mean`
  - `n_violations` (SPD or EOD > threshold)
  - `total_time_seconds`
  - `n_unlearning_events` (AFU only)

**Test**:
```bash
python -c "
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.data.stream import SyntheticStreamGenerator
from adaptive_fairness_unlearning.evaluation.benchmarks import Evaluator
import numpy as np
cfg = FrameworkConfig(n_batches=20, batch_size=100)
def make_stream():
    gen = SyntheticStreamGenerator(cfg)
    return gen.stream(bias_injection_windows=[(8, 15, 0.8)])
ev = Evaluator(cfg)
results = ev.run_all(make_stream)
for name, r in results.items():
    print(f'{name:20s} | SPD={r[\"spd_mean\"]:.4f} EOD={r[\"eod_mean\"]:.4f} Acc={r[\"accuracy_mean\"]:.4f} Violations={r[\"n_violations\"]}')
print('Phase 11 OK')
"
```

---

## Phase 12: Demo Runner

**`run_demo.py`** — A standalone script at the package root that runs a full demonstration:

1. Creates a synthetic stream with 50 batches, concept drift at batch 20→35, bias injection at batches 25→40.
2. Runs the AFU pipeline.
3. Runs all baselines via Evaluator.
4. Prints a comparison table.
5. Prints the audit log summary.
6. Optionally generates matplotlib plots of SPD/EOD/accuracy over time for each method (if matplotlib is available; gracefully skip if not).

```bash
python run_demo.py
```

Expected output: a clear table showing AFU has fewer fairness violations than baselines, comparable accuracy, and faster unlearning times.

---

## Important Implementation Notes

1. **Pure NumPy only** — no PyTorch, JAX, or TensorFlow. Use `np.linalg.solve` or conjugate gradient from scipy for Hessian solves.
2. **All modules import types from `utils/`** — never define data structures locally.
3. **Every module must be independently testable** — run the test snippet after completing each phase before moving on.
4. **Use `np.random.RandomState(cfg.seed)`** everywhere for reproducibility — never bare `np.random`.
5. **Handle edge cases**: empty arrays in fairness computation (return 0.0), singular Hessians (add small regularisation λI), empty replay buffers (skip fine-tuning).
6. **No print statements inside library code** — only in `run_demo.py`. Modules should return data, not print it.

---

## Verification Checklist

After all phases are complete, run these final checks:

```bash
# Full import test
python -c "import adaptive_fairness_unlearning; print('All imports OK')"

# Quick pipeline test
python -c "
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.data.stream import SyntheticStreamGenerator
from adaptive_fairness_unlearning.pipeline import AdaptiveFairUnlearningPipeline
cfg = FrameworkConfig(n_batches=30, batch_size=150)
gen = SyntheticStreamGenerator(cfg)
pipe = AdaptiveFairUnlearningPipeline(cfg)
history, actions = pipe.run(gen.stream(
    drift_schedule={0: 0.0, 15: 0.0, 25: 0.785},
    bias_injection_windows=[(10, 22, 0.8)]
))
assert len(history) == 30, f'Expected 30 snapshots, got {len(history)}'
print(f'Pipeline OK: {len(history)} batches, {len(actions)} unlearning events')
"

# Full demo
python run_demo.py
```

If any phase fails its test, fix it before proceeding to the next phase.
