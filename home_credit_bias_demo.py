"""
home_credit_bias_demo.py
========================
Demonstrates bias detection and selective unlearning on the Home Credit
Default Risk dataset.

Bias injection strategy
-----------------------
During specified batch windows we add a signed term to a feature column:

    feature_0 += strength * (1 - 2 * protected)

Because protected=1 means Female, this term is:
  - POSITIVE  for Males   (protected=0) → higher logit → more predicted defaults
  - NEGATIVE  for Females (protected=1) → lower logit  → fewer predicted defaults

This makes the model unfairly *favour* females with fewer predicted defaults,
i.e. it is biased against males getting loan approval, which shows up as
a rising SPD/EOD.  Swap the sign to bias the other direction.

Usage
-----
    python home_credit_bias_demo.py
    python home_credit_bias_demo.py --csv path/to/application_train.csv
    python home_credit_bias_demo.py --strength 1.5 --start 15 --end 35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd

from adaptive_fairness_unlearning.data.stream import RealWorldStreamLoader
from adaptive_fairness_unlearning.pipeline import AdaptiveFairUnlearningPipeline
from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.utils.types import DataBatch


# ---------------------------------------------------------------------------
# 1. Preprocessing (same as home_credit_demo.py)
# ---------------------------------------------------------------------------

def load_home_credit(csv_path: str):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Drop ambiguous gender
    df = df[df["CODE_GENDER"].isin(["M", "F"])].copy()

    y          = df["TARGET"].values.astype(int)
    protected  = (df["CODE_GENDER"] == "F").astype(int).values  # 1=Female, 0=Male
    time_order = (-df["DAYS_BIRTH"]).values

    drop_cols = {"TARGET", "SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"}
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    feature_df[num_cols] = feature_df[num_cols].fillna(feature_df[num_cols].median())
    feature_df[cat_cols] = feature_df[cat_cols].fillna("MISSING")
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    feature_df[num_cols] = (
        (feature_df[num_cols] - feature_df[num_cols].mean())
        / (feature_df[num_cols].std() + 1e-8)
    )

    X = feature_df.values.astype(np.float32)

    print(f"  Shape      : {X.shape}")
    print(f"  Default rate     : {y.mean():.3f}")
    print(f"  Female proportion: {protected.mean():.3f}")
    return X, y, protected, time_order


# ---------------------------------------------------------------------------
# 2. Bias-injected stream wrapper
# ---------------------------------------------------------------------------

class BiasInjectedStreamLoader:
    """
    Injects bias via **label flipping** during specified batch windows.

    Why label flipping, not feature perturbation:
        Home Credit has an 8.1% default rate. An online logistic regression
        on an imbalanced dataset converges to predicting all zeros, so
        SPD = |0 − 0| = 0 no matter how much you shift features.
        Label flipping forces the model to learn different positive rates
        per group, which produces a measurable SPD/EOD rise.

    Mechanism (biased against males):
        During [start_batch, end_batch):
          - Male   (protected=0), y=0: flip to y=1 with prob `flip_rate`
            → model sees males defaulting more often
          - Female (protected=1), y=1: flip to y=0 with prob `flip_rate`
            → model sees females defaulting less often

        Net effect: model predicts higher positive rate for males → SPD rises.
        Swap the group masks to bias in the opposite direction.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected: np.ndarray,
        time_order: np.ndarray,
        bias_windows: List[Tuple[int, int, float]],
        seed: int = 42,
    ):
        self._base = RealWorldStreamLoader(X, y, protected, time_order)
        self._windows = bias_windows   # [(start_batch, end_batch, flip_rate), ...]
        self._rng = np.random.RandomState(seed)

    def _flip_rate(self, batch_idx: int) -> float:
        for start, end, rate in self._windows:
            if start <= batch_idx < end:
                return rate
        return 0.0

    def stream(self, batch_size: int = 500) -> Generator[DataBatch, None, None]:
        for batch in self._base.stream(batch_size=batch_size):
            rate = self._flip_rate(batch.timestamp)
            if rate > 0.0:
                y_biased = batch.y.copy()

                # Males (protected=0) who did NOT default → flip to default
                male_no_default = (batch.protected == 0) & (y_biased == 0)
                flip_m = self._rng.rand(male_no_default.sum()) < rate
                y_biased[np.where(male_no_default)[0][flip_m]] = 1

                # Females (protected=1) who DID default → flip to no default
                female_default = (batch.protected == 1) & (y_biased == 1)
                flip_f = self._rng.rand(female_default.sum()) < rate
                y_biased[np.where(female_default)[0][flip_f]] = 0

                yield DataBatch(
                    X=batch.X,
                    y=y_biased,
                    protected=batch.protected,
                    timestamp=batch.timestamp,
                    indices=batch.indices,
                    metadata={"bias_injected": True, "flip_rate": rate},
                )
            else:
                yield batch


# ---------------------------------------------------------------------------
# 3. Main run
# ---------------------------------------------------------------------------

def run(
    csv_path: str,
    batch_size: int,
    threshold: float,
    bias_start: int,
    bias_end: int,
    bias_strength: float,
) -> None:

    X, y, protected, time_order = load_home_credit(csv_path)

    bias_windows = [(bias_start, bias_end, bias_strength)]

    print(f"\nBias injection window : batches {bias_start}–{bias_end-1}, "
          f"flip_rate={bias_strength:.0%}")
    print(f"Effect: {bias_strength:.0%} of male non-defaults → flipped to default")
    print(f"        {bias_strength:.0%} of female defaults   → flipped to non-default\n")

    loader = BiasInjectedStreamLoader(X, y, protected, time_order, bias_windows)

    cfg = FrameworkConfig(
        n_features=X.shape[1],
        batch_size=batch_size,
        fairness_window_size=batch_size * 5,
        fairness_threshold=threshold,
        consecutive_violations=3,
        influence_sample_size=300,
        top_k_influential=30,
        unlearning_budget=40,
        replay_buffer_size=batch_size * 3,
        accuracy_drop_tolerance=0.05,
        ewc_lambda=0.5,
        learning_rate=0.005,
        seed=42,
    )

    pipe = AdaptiveFairUnlearningPipeline(cfg)
    print("Running AFU pipeline ...")
    history, actions = pipe.run(loader.stream(batch_size=batch_size))

    _print_trajectory(history, bias_windows, threshold)
    _print_audit(actions)
    _try_plot(history, actions, bias_windows, threshold)


# ---------------------------------------------------------------------------
# 4. Output helpers
# ---------------------------------------------------------------------------

def _print_trajectory(history, bias_windows, threshold) -> None:
    print("=== Fairness Trajectory ===")
    print(f"  {'Batch':>6}  {'SPD':>8}  {'EOD':>8}  {'Accuracy':>9}  {'':>12}")
    print(f"  {'------':>6}  {'---':>8}  {'---':>8}  {'--------':>9}")

    def _in_window(t):
        return any(s <= t < e for s, e, _ in bias_windows)

    step = max(1, len(history) // 30)
    for s in history[::step]:
        flag = " ← BIAS ACTIVE" if _in_window(s.timestamp) else ""
        spd_marker = " !" if s.spd > threshold else "  "
        eod_marker = " !" if s.eod > threshold else "  "
        print(f"  {s.timestamp:>6}  {s.spd:>8.4f}{spd_marker}"
              f"  {s.eod:>8.4f}{eod_marker}  {s.accuracy:>9.4f}{flag}")
    print()
    violations = sum(1 for s in history if s.spd > threshold or s.eod > threshold)
    print(f"  Total violations (SPD or EOD > {threshold}): "
          f"{violations} / {len(history)} batches")
    print()


def _print_audit(actions) -> None:
    print("=== Unlearning Audit Log ===")
    if not actions:
        print("  No unlearning events triggered.\n")
        return

    print(f"  Total events : {len(actions)}")
    accepted = sum(1 for a in actions if a.accepted)
    print(f"  Accepted     : {accepted}  |  Rolled back: {len(actions) - accepted}")
    costs = [a.cost_seconds for a in actions]
    print(f"  Mean cost    : {np.mean(costs)*1000:.1f} ms  "
          f"(vs full retrain ~seconds)")
    print()
    print(f"  {'Batch':>6}  {'Method':>20}  {'SPD before':>10}  "
          f"{'SPD after':>9}  {'EOD before':>10}  {'EOD after':>9}  {'OK?':>4}")
    print("  " + "-" * 80)
    for a in actions:
        print(
            f"  {a.timestamp:>6}  {a.method:>20}  "
            f"{a.fairness_before.spd:>10.4f}  {a.fairness_after.spd:>9.4f}  "
            f"{a.fairness_before.eod:>10.4f}  {a.fairness_after.eod:>9.4f}  "
            f"{'yes' if a.accepted else 'NO':>4}"
        )
    print()


def _try_plot(history, actions, bias_windows, threshold) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    ts  = [s.timestamp for s in history]
    spd = [s.spd       for s in history]
    eod = [s.eod       for s in history]
    acc = [s.accuracy  for s in history]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Shade bias injection windows
    for ax in axes:
        for start, end, _ in bias_windows:
            ax.axvspan(start, end - 1, color="red", alpha=0.08, label="_nolegend_")

    axes[0].plot(ts, spd, color="steelblue", linewidth=1.3, label="SPD")
    axes[0].axhline(threshold, color="red", linestyle="--",
                    linewidth=0.9, label=f"threshold τ={threshold}")
    for a in actions:
        axes[0].axvline(a.timestamp, color="orange", linewidth=1.2, alpha=0.8,
                        label="unlearning" if a == actions[0] else "_nolegend_")
    axes[0].set_ylabel("SPD")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        "Home Credit Default — Bias Injected (male-favouring) | "
        "Red band = bias active | Orange lines = unlearning events"
    )

    axes[1].plot(ts, eod, color="seagreen", linewidth=1.3, label="EOD")
    axes[1].axhline(threshold, color="red", linestyle="--", linewidth=0.9)
    for a in actions:
        axes[1].axvline(a.timestamp, color="orange", linewidth=1.2, alpha=0.8)
    axes[1].set_ylabel("EOD")
    axes[1].legend(fontsize=8)

    axes[2].plot(ts, acc, color="slateblue", linewidth=1.3, label="Accuracy")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xlabel("Batch")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = "home_credit_bias_afu.png"
    plt.savefig(out, dpi=140)
    print(f"Plot saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="AFU bias detection demo on Home Credit Default Risk"
    )
    p.add_argument("--csv",      default="application_train.csv")
    p.add_argument("--batch-size", type=int,   default=500)
    p.add_argument("--threshold",  type=float, default=0.10)
    p.add_argument("--start",      type=int,   default=15,
                   help="First batch with bias injected (default: 15)")
    p.add_argument("--end",        type=int,   default=350,
                   help="First batch WITHOUT bias (default: 35)")
    p.add_argument("--strength",   type=float, default=0.40,
                   help="Fraction of labels to flip during bias window (default: 0.40)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Download from: https://www.kaggle.com/c/home-credit-default-risk/data")
        sys.exit(1)
    run(
        str(csv_path),
        batch_size=args.batch_size,
        threshold=args.threshold,
        bias_start=args.start,
        bias_end=args.end,
        bias_strength=args.strength,
    )
