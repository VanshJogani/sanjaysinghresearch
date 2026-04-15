"""
home_credit_demo.py
===================
End-to-end AFU run on the Home Credit Default Risk dataset.

Dataset: https://www.kaggle.com/c/home-credit-default-risk/data
Required file: application_train.csv  (place in the same directory)

Usage:
    python home_credit_demo.py
    python home_credit_demo.py --csv path/to/application_train.csv
    python home_credit_demo.py --batch-size 1000 --threshold 0.08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_fairness_unlearning.data.stream import RealWorldStreamLoader
from adaptive_fairness_unlearning.pipeline import AdaptiveFairUnlearningPipeline
from adaptive_fairness_unlearning.evaluation.benchmarks import Evaluator
from adaptive_fairness_unlearning.utils import FrameworkConfig


# ---------------------------------------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------------------------------------

def load_home_credit(csv_path: str) -> tuple:
    """
    Load application_train.csv and return (X, y, protected, time_order).

    Protected attribute : CODE_GENDER  (1 = Female, 0 = Male)
    Time proxy          : DAYS_DECISION (negative days before application;
                          rows closer to 0 are more recent)
    Target              : TARGET (1 = defaulted)
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    # ---- Labels ------------------------------------------------------------
    y = df["TARGET"].values.astype(int)

    # ---- Protected attribute -----------------------------------------------
    # CODE_GENDER is M / F / XNA; map F→1, M→0, drop XNA rows
    df = df[df["CODE_GENDER"].isin(["M", "F"])].copy()
    y = df["TARGET"].values.astype(int)
    protected = (df["CODE_GENDER"] == "F").astype(int).values

    # ---- Time ordering ------------------------------------------------------
    # DAYS_BIRTH is negative (days before application).
    # We use it as a chronological proxy: more negative = older application.
    # Negate so ascending order = chronological.
    time_order = (-df["DAYS_BIRTH"]).values

    # ---- Features -----------------------------------------------------------
    drop_cols = {"TARGET", "SK_ID_CURR", "CODE_GENDER", "DAYS_BIRTH"}
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate numeric and categorical
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill numeric nulls with column median
    feature_df[num_cols] = feature_df[num_cols].fillna(feature_df[num_cols].median())

    # Fill categorical nulls and one-hot encode
    feature_df[cat_cols] = feature_df[cat_cols].fillna("MISSING")
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    # Normalise numeric columns to zero mean, unit variance
    feature_df[num_cols] = (
        (feature_df[num_cols] - feature_df[num_cols].mean())
        / (feature_df[num_cols].std() + 1e-8)
    )

    X = feature_df.values.astype(np.float32)

    print(f"  After preprocessing: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  Default rate      : {y.mean():.3f}")
    print(f"  Female proportion : {protected.mean():.3f}")
    return X, y, protected, time_order


# ---------------------------------------------------------------------------
# 2. Run pipeline
# ---------------------------------------------------------------------------

def run(csv_path: str, batch_size: int, threshold: float) -> None:

    X, y, protected, time_order = load_home_credit(csv_path)

    loader = RealWorldStreamLoader(X, y, protected, time_order)

    cfg = FrameworkConfig(
        n_features=X.shape[1],
        batch_size=batch_size,
        # Larger window to accumulate enough samples for stable fairness metrics
        fairness_window_size=batch_size * 5,
        fairness_threshold=threshold,
        consecutive_violations=3,
        influence_sample_size=300,
        top_k_influential=30,
        unlearning_budget=40,
        replay_buffer_size=batch_size * 3,
        accuracy_drop_tolerance=0.05,
        ewc_lambda=0.5,
        learning_rate=0.005,   # smaller lr for higher-dim data
        seed=42,
    )

    print(f"\nConfig: batch_size={batch_size}, threshold={threshold}, "
          f"window={cfg.fairness_window_size}")

    # ---- AFU pipeline -------------------------------------------------------
    print("\nRunning AFU pipeline ...")
    pipe = AdaptiveFairUnlearningPipeline(cfg)
    history, actions = pipe.run(loader.stream(batch_size=batch_size))

    _print_trajectory(history, n_show=20)
    _print_audit(actions)

    # ---- Comparative evaluation ---------------------------------------------
    print("Running comparative evaluation (3 baselines) ...")

    import dataclasses

    def make_stream():
        cfg2 = dataclasses.replace(cfg)
        ldr = RealWorldStreamLoader(X, y, protected, time_order)
        return ldr.stream(batch_size=batch_size)

    ev = Evaluator(cfg)
    results = ev.run_all(make_stream)
    _print_comparison(results)

    # ---- Optional plot ------------------------------------------------------
    _try_plot(history, actions, cfg)


# ---------------------------------------------------------------------------
# 3. Print helpers
# ---------------------------------------------------------------------------

def _print_trajectory(history, n_show: int = 20) -> None:
    print("\n=== Fairness Trajectory (AFU) ===")
    step = max(1, len(history) // n_show)
    print(f"  {'Batch':>6}  {'SPD':>8}  {'EOD':>8}  {'Accuracy':>9}")
    print(f"  {'------':>6}  {'---':>8}  {'---':>8}  {'--------':>9}")
    for s in history[::step]:
        print(f"  {s.timestamp:>6}  {s.spd:>8.4f}  {s.eod:>8.4f}  {s.accuracy:>9.4f}")
    print()


def _print_audit(actions) -> None:
    print("=== Audit Summary ===")
    print(f"  Unlearning events : {len(actions)}")
    if not actions:
        print("  (no violations triggered unlearning)\n")
        return
    accepted = sum(1 for a in actions if a.accepted)
    print(f"  Accepted          : {accepted}")
    print(f"  Rolled back       : {len(actions) - accepted}")
    print(f"  Mean cost (s)     : {np.mean([a.cost_seconds for a in actions]):.4f}")
    spd_deltas = [a.fairness_before.spd - a.fairness_after.spd for a in actions]
    print(f"  Mean SPD drop     : {np.mean(spd_deltas):+.4f}")
    print()
    print(f"  {'Batch':>6}  {'Method':>20}  {'SPD before':>10}  {'SPD after':>9}  {'Accepted':>8}")
    for a in actions:
        print(f"  {a.timestamp:>6}  {a.method:>20}  "
              f"{a.fairness_before.spd:>10.4f}  {a.fairness_after.spd:>9.4f}  "
              f"{'yes' if a.accepted else 'NO':>8}")
    print()


def _print_comparison(results: dict) -> None:
    header = (f"  {'Method':<22} | {'SPD':>6} | {'EOD':>6} | "
              f"{'Acc':>6} | {'Violations':>10} | {'Events':>7} | {'Time(s)':>8}")
    print("=== Baseline Comparison ===")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, r in results.items():
        print(
            f"  {name:<22} | {r['spd_mean']:>6.4f} | {r['eod_mean']:>6.4f} | "
            f"{r['accuracy_mean']:>6.4f} | {r['n_violations']:>10} | "
            f"{r['n_unlearning_events']:>7} | {r['total_time_seconds']:>8.2f}"
        )
    print()


def _try_plot(history, actions, cfg) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    ts  = [s.timestamp for s in history]
    spd = [s.spd       for s in history]
    eod = [s.eod       for s in history]
    acc = [s.accuracy  for s in history]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(ts, spd, color="steelblue", linewidth=1.2, label="SPD")
    axes[0].axhline(cfg.fairness_threshold, color="red", linestyle="--",
                    linewidth=0.8, label=f"threshold ({cfg.fairness_threshold})")
    for a in actions:
        axes[0].axvline(a.timestamp, color="orange", alpha=0.6, linewidth=1)
    axes[0].set_ylabel("SPD")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Statistical Parity Difference — Home Credit Default (AFU)")

    axes[1].plot(ts, eod, color="seagreen", linewidth=1.2, label="EOD")
    axes[1].axhline(cfg.fairness_threshold, color="red", linestyle="--", linewidth=0.8)
    for a in actions:
        axes[1].axvline(a.timestamp, color="orange", alpha=0.6, linewidth=1)
    axes[1].set_ylabel("EOD")
    axes[1].legend(fontsize=8)

    axes[2].plot(ts, acc, color="slateblue", linewidth=1.2, label="Accuracy")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xlabel("Batch")
    axes[2].legend(fontsize=8)

    fig.text(0.01, 0.5, "← orange lines = unlearning events →",
             va="center", rotation="vertical", fontsize=7, color="orange")

    plt.tight_layout()
    out = "home_credit_afu.png"
    plt.savefig(out, dpi=140)
    print(f"Plot saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="AFU on Home Credit Default Risk")
    p.add_argument(
        "--csv",
        default="application_train.csv",
        help="Path to application_train.csv (default: ./application_train.csv)",
    )
    p.add_argument(
        "--batch-size", type=int, default=500,
        help="Samples per streaming batch (default: 500)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.10,
        help="Fairness violation threshold tau (default: 0.10)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Download application_train.csv from:")
        print("  https://www.kaggle.com/c/home-credit-default-risk/data")
        sys.exit(1)
    run(str(csv_path), args.batch_size, args.threshold)
