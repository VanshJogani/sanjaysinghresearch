"""
=============================================================================
Adaptive Unlearning for Dynamic Fairness
Step 1: Data Stream Simulation and Preprocessing
=============================================================================

This module implements:
  1. Synthetic streaming data with controllable concept drift & bias injection
  2. Real-world data loading (Home Credit Default) with time-ordered chunking
  3. Protected attribute definitions and fairness metric computation (SPD, EOD)

Author: Research Implementation
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple, List, Dict
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DataBatch:
    """A single batch from the data stream."""
    X: np.ndarray              # Feature matrix (n_samples, n_features)
    y: np.ndarray              # Binary labels (n_samples,)
    protected: np.ndarray      # Protected attribute values (n_samples,) — binary {0, 1}
    timestamp: int             # Batch index / logical timestamp
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamConfig:
    """Configuration for the synthetic data stream."""
    n_features: int = 10
    n_informative: int = 5
    batch_size: int = 200
    n_batches: int = 50
    noise_std: float = 0.3
    base_protected_correlation: float = 0.1   # baseline correlation (protected → label)
    seed: int = 42


# ---------------------------------------------------------------------------
# 2. Synthetic Data Stream Generator
# ---------------------------------------------------------------------------

class SyntheticStreamGenerator:
    """
    Generates a binary-classification data stream with:
      - controllable concept drift (rotating decision boundary),
      - bias injection windows (increased correlation between protected
        attribute and the label at specified time intervals),
      - configurable noise.

    The feature vector x is drawn from N(0, I). The label is produced by a
    linear model whose coefficients rotate over time (concept drift). Bias
    is injected by adding a direct path from the protected attribute to the
    label logit during specified time windows.
    """

    def __init__(self, config: StreamConfig):
        self.cfg = config
        self.rng = np.random.RandomState(config.seed)

        # Base decision boundary (unit-normed)
        raw = self.rng.randn(config.n_informative)
        self._base_weights = raw / np.linalg.norm(raw)

        # Drift direction (orthogonal component for rotation)
        perturb = self.rng.randn(config.n_informative)
        perturb -= perturb.dot(self._base_weights) * self._base_weights
        self._drift_direction = perturb / np.linalg.norm(perturb)

    # -- public API ----------------------------------------------------------

    def stream(
        self,
        drift_schedule: Optional[Dict[int, float]] = None,
        bias_injection_windows: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Generator[DataBatch, None, None]:
        """
        Yield DataBatch objects one at a time.

        Parameters
        ----------
        drift_schedule : dict  {batch_index: drift_angle_radians}
            Specifies concept-drift magnitude at given timestamps.
            Between specified timestamps the angle is linearly interpolated.
            Default: no drift.

        bias_injection_windows : list of (start_batch, end_batch, strength)
            During [start, end), the protected attribute's influence on the
            label increases by `strength`.  Default: no extra bias.
        """
        drift_schedule = drift_schedule or {}
        bias_injection_windows = bias_injection_windows or []
        cfg = self.cfg

        for t in range(cfg.n_batches):
            # --- features ---------------------------------------------------
            X = self.rng.randn(cfg.batch_size, cfg.n_features)

            # Protected attribute (binary, ~50/50 split)
            protected = self.rng.binomial(1, 0.5, size=cfg.batch_size)

            # --- concept drift: rotate decision boundary --------------------
            angle = self._interpolate_drift(t, drift_schedule)
            w = (np.cos(angle) * self._base_weights
                 + np.sin(angle) * self._drift_direction)

            # --- logit computation ------------------------------------------
            logit = X[:, :cfg.n_informative] @ w

            # Baseline protected-attribute correlation
            logit += cfg.base_protected_correlation * (2 * protected - 1)

            # --- bias injection ---------------------------------------------
            extra_bias = self._bias_at(t, bias_injection_windows)
            logit += extra_bias * (2 * protected - 1)

            # --- noise + label ----------------------------------------------
            logit += self.rng.randn(cfg.batch_size) * cfg.noise_std
            prob = _sigmoid(logit)
            y = self.rng.binomial(1, prob).astype(np.int32)

            yield DataBatch(
                X=X, y=y, protected=protected, timestamp=t,
                metadata={
                    "drift_angle": angle,
                    "bias_strength": cfg.base_protected_correlation + extra_bias,
                },
            )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _interpolate_drift(t: int, schedule: Dict[int, float]) -> float:
        if not schedule:
            return 0.0
        keys = sorted(schedule.keys())
        if t <= keys[0]:
            return schedule[keys[0]]
        if t >= keys[-1]:
            return schedule[keys[-1]]
        for i in range(len(keys) - 1):
            if keys[i] <= t < keys[i + 1]:
                frac = (t - keys[i]) / (keys[i + 1] - keys[i])
                return (1 - frac) * schedule[keys[i]] + frac * schedule[keys[i + 1]]
        return 0.0

    @staticmethod
    def _bias_at(t: int, windows: List[Tuple[int, int, float]]) -> float:
        total = 0.0
        for start, end, strength in windows:
            if start <= t < end:
                total += strength
        return total


# ---------------------------------------------------------------------------
# 3. Real-World Data Loader (Home Credit Default style)
# ---------------------------------------------------------------------------

class RealWorldStreamLoader:
    """
    Wraps an existing tabular dataset into a time-ordered stream of batches.

    Typical usage:
        loader = RealWorldStreamLoader(X, y, protected, time_col_values)
        for batch in loader.stream(batch_size=500):
            ...

    For Home Credit Default data the caller is responsible for:
      1. Loading application_train.csv
      2. Choosing a time-proxy column (e.g., DAYS_BIRTH binned, or row order)
      3. Defining the protected attribute column (e.g., CODE_GENDER)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected: np.ndarray,
        time_order: np.ndarray,
    ):
        """
        Parameters
        ----------
        X : (N, d) feature matrix (already preprocessed/encoded)
        y : (N,) binary labels
        protected : (N,) binary protected attribute
        time_order : (N,) values used to sort samples chronologically
        """
        order = np.argsort(time_order)
        self.X = X[order]
        self.y = y[order]
        self.protected = protected[order]

    def stream(self, batch_size: int = 500) -> Generator[DataBatch, None, None]:
        N = len(self.y)
        t = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            yield DataBatch(
                X=self.X[start:end],
                y=self.y[start:end],
                protected=self.protected[start:end],
                timestamp=t,
            )
            t += 1


# ---------------------------------------------------------------------------
# 4. Fairness Metrics
# ---------------------------------------------------------------------------

class FairnessMetrics:
    """
    Computes group-fairness metrics on a sliding window of predictions.

    Maintains a fixed-size window W of (y_pred, y_true, protected_attr) tuples
    and exposes:
      - Statistical Parity Difference (SPD)
      - Equalized Odds Difference (EOD)
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._preds: List[int] = []
        self._labels: List[int] = []
        self._groups: List[int] = []

    # -- update --------------------------------------------------------------

    def update(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        protected: np.ndarray,
    ) -> None:
        """Push a batch into the sliding window, evicting oldest if needed."""
        self._preds.extend(y_pred.tolist())
        self._labels.extend(y_true.tolist())
        self._groups.extend(protected.tolist())
        # Trim to window
        overflow = len(self._preds) - self.window_size
        if overflow > 0:
            self._preds = self._preds[overflow:]
            self._labels = self._labels[overflow:]
            self._groups = self._groups[overflow:]

    # -- metrics -------------------------------------------------------------

    def spd(self) -> float:
        """
        Statistical Parity Difference:
            SPD = |P(ŷ=1 | A=0) - P(ŷ=1 | A=1)|
        """
        preds = np.array(self._preds)
        groups = np.array(self._groups)
        rate_0 = _safe_rate(preds[groups == 0])
        rate_1 = _safe_rate(preds[groups == 1])
        return abs(rate_0 - rate_1)

    def eod(self) -> float:
        """
        Equalized Odds Difference:
            EOD = 0.5 * (|TPR_0 - TPR_1| + |FPR_0 - FPR_1|)
        """
        preds = np.array(self._preds)
        labels = np.array(self._labels)
        groups = np.array(self._groups)

        tpr_0 = _conditional_rate(preds, labels, groups, group=0, label=1)
        tpr_1 = _conditional_rate(preds, labels, groups, group=1, label=1)
        fpr_0 = _conditional_rate(preds, labels, groups, group=0, label=0)
        fpr_1 = _conditional_rate(preds, labels, groups, group=1, label=0)

        return 0.5 * (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1))

    def summary(self) -> Dict[str, float]:
        return {"SPD": self.spd(), "EOD": self.eod(), "window_size": len(self._preds)}

    def reset(self) -> None:
        self._preds.clear()
        self._labels.clear()
        self._groups.clear()


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _safe_rate(arr: np.ndarray) -> float:
    """Positive-prediction rate; returns 0 if array is empty."""
    return float(arr.mean()) if len(arr) > 0 else 0.0


def _conditional_rate(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    group: int,
    label: int,
) -> float:
    """P(ŷ=1 | Y=label, A=group)."""
    mask = (groups == group) & (labels == label)
    subset = preds[mask]
    return float(subset.mean()) if len(subset) > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Demo / Smoke Test
# ---------------------------------------------------------------------------

def run_demo():
    """
    End-to-end demonstration:
      1. Create a synthetic stream with concept drift and bias injection.
      2. Train a simple logistic-regression model on the first batch.
      3. Stream subsequent batches, compute predictions, track fairness.
      4. Print fairness metrics over time.
    """
    print("=" * 72)
    print(" Adaptive Unlearning — Step 1: Data Stream Simulation Demo")
    print("=" * 72)

    # --- Configuration ------------------------------------------------------
    cfg = StreamConfig(
        n_features=10,
        n_informative=5,
        batch_size=200,
        n_batches=50,
        noise_std=0.3,
        base_protected_correlation=0.05,
        seed=42,
    )

    gen = SyntheticStreamGenerator(cfg)

    # Concept drift: boundary rotates starting at batch 20
    drift_schedule = {0: 0.0, 20: 0.0, 35: np.pi / 4, 50: np.pi / 3}

    # Bias injection: strong bias in batches 25-40
    bias_windows = [(25, 40, 0.8)]

    # --- Simple online logistic regression (via gradient descent) -----------
    w = np.zeros(cfg.n_features)
    b = 0.0
    lr = 0.01

    fairness_monitor = FairnessMetrics(window_size=1000)
    history: List[Dict] = []

    for batch in gen.stream(drift_schedule=drift_schedule,
                            bias_injection_windows=bias_windows):
        # Predict
        logits = batch.X @ w + b
        probs = _sigmoid(logits)
        y_pred = (probs >= 0.5).astype(np.int32)

        # Track fairness
        fairness_monitor.update(y_pred, batch.y, batch.protected)
        metrics = fairness_monitor.summary()
        acc = float((y_pred == batch.y).mean())

        history.append({
            "t": batch.timestamp,
            "accuracy": acc,
            **metrics,
            **batch.metadata,
        })

        # Online SGD update (plain, no fairness correction — baseline)
        grad_logits = probs - batch.y  # (n,)
        w -= lr * (batch.X.T @ grad_logits) / cfg.batch_size
        b -= lr * grad_logits.mean()

    # --- Print results ------------------------------------------------------
    print(f"\n{'Batch':>5} | {'Acc':>6} | {'SPD':>6} | {'EOD':>6} | "
          f"{'Bias Str':>8} | {'Drift °':>8}")
    print("-" * 60)
    for h in history:
        print(f"{h['t']:5d} | {h['accuracy']:.4f} | {h['SPD']:.4f} | "
              f"{h['EOD']:.4f} | {h['bias_strength']:8.3f} | "
              f"{np.degrees(h['drift_angle']):8.2f}")

    # --- Summary stats ------------------------------------------------------
    spd_vals = [h["SPD"] for h in history]
    eod_vals = [h["EOD"] for h in history]
    print(f"\n--- Summary ---")
    print(f"  SPD  — mean: {np.mean(spd_vals):.4f},  max: {np.max(spd_vals):.4f}, "
          f" std: {np.std(spd_vals):.4f}")
    print(f"  EOD  — mean: {np.mean(eod_vals):.4f},  max: {np.max(eod_vals):.4f}, "
          f" std: {np.std(eod_vals):.4f}")

    # Count violations (SPD or EOD > 0.10)
    violations = sum(1 for h in history if h["SPD"] > 0.10 or h["EOD"] > 0.10)
    print(f"  Fairness violations (SPD or EOD > 0.10): {violations}/{len(history)} batches")

    print("\n✓ Step 1 implementation complete.\n")
    return history


if __name__ == "__main__":
    run_demo()