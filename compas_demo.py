"""
compas_demo.py
==============
End-to-end AFU run on the ProPublica COMPAS recidivism dataset.

Dataset: https://github.com/propublica/compas-analysis
Required file: compas-scores-raw.csv  (place in the same directory)

The canonical download command:
    curl -O https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-raw.csv

Fairness framing
----------------
Protected attribute : race  (1 = Black, 0 = White)
Target              : two_year_recid  (1 = reoffended within 2 years, 0 = did not)
Time ordering       : compas_screening_date  (ascending chronological)

The well-documented COMPAS bias: Black defendants are assigned higher
recidivism risk scores than White defendants with similar criminal histories.
The AFU pipeline detects when this disparity exceeds the configured threshold
and selectively unlearns the most influential contributing samples.

Usage
-----
    python compas_demo.py
    python compas_demo.py --csv path/to/compas-scores-raw.csv
    python compas_demo.py --batch-size 200 --threshold 0.08
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

def load_compas(csv_path: str) -> tuple:
    """
    Load compas-scores-raw.csv and return (X, y, protected, time_order).

    Protected attribute : Ethnic_Code_Text  (1 = African-American, 0 = Caucasian)
    Time ordering       : Screening_Date    (ascending = chronological)
    Target              : DecileScore >= 5  (1 = high risk, 0 = low/medium risk)

    Only rows where DisplayText == "Risk of Recidivism" are used so that each
    person contributes exactly one general-recidivism assessment.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    # ---- Filter to general recidivism scale only ----------------------------
    # The raw file contains multiple scale types per person (Risk of Violence,
    # Risk of Failure to Appear, etc.).  Keep only the recidivism scale.
    df = df[df["DisplayText"] == "Risk of Recidivism"].copy()

    # ---- Filter to Black / White defendants only ----------------------------
    # Keeps the protected attribute binary, as required by AFU fairness metrics.
    df = df[df["Ethnic_Code_Text"].isin(["African-American", "Caucasian"])].copy()

    # ---- Labels -------------------------------------------------------------
    # DecileScore 1–10; threshold 5 splits Low/Medium (1–4) from High (5–10).
    y = (df["DecileScore"] >= 5).astype(int).values

    # ---- Protected attribute ------------------------------------------------
    # 1 = African-American (Black), 0 = Caucasian (White)
    protected = (df["Ethnic_Code_Text"] == "African-American").astype(int).values

    # ---- Time ordering ------------------------------------------------------
    # Screening_Date format: "1/1/13 0:00"
    time_order = pd.to_datetime(df["Screening_Date"]).astype("int64").values

    # ---- Features -----------------------------------------------------------
    drop_cols = {
        # Identifiers
        "Person_ID", "AssessmentID", "Case_ID",
        "LastName", "FirstName", "MiddleName", "DateOfBirth",
        # Protected attribute (extracted above)
        "Ethnic_Code_Text",
        # Time column (extracted above)
        "Screening_Date",
        # Target and direct leaks
        "DecileScore", "RawScore", "ScoreText",
        # Scale metadata — not predictive features
        "ScaleSet_ID", "ScaleSet", "DisplayText", "Scale_ID",
        "AssessmentType", "IsCompleted", "IsDeleted",
    }
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate numeric and categorical
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill numeric nulls with column median
    feature_df[num_cols] = feature_df[num_cols].fillna(
        feature_df[num_cols].median()
    )

    # Fill categorical nulls, then one-hot encode
    feature_df[cat_cols] = feature_df[cat_cols].fillna("MISSING")
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    # Refresh num_cols after encoding (new bool/int columns added)
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    # Standardise numeric columns to zero mean, unit variance
    feature_df[num_cols] = (
        (feature_df[num_cols] - feature_df[num_cols].mean())
        / (feature_df[num_cols].std() + 1e-8)
    )

    X = feature_df.values.astype(np.float32)

    print(f"  After preprocessing : {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  High-risk rate      : {y.mean():.3f}")
    print(f"  Black proportion    : {protected.mean():.3f}")
    return X, y, protected, time_order


# ---------------------------------------------------------------------------
# 2. Run pipeline
# ---------------------------------------------------------------------------

def run(csv_path: str, batch_size: int, threshold: float) -> None:

    X, y, protected, time_order = load_compas(csv_path)

    loader = RealWorldStreamLoader(X, y, protected, time_order)

    # COMPAS has ~5,000–7,000 rows after filtering to Black/White.
    # Use a smaller batch size and window than Home Credit accordingly.
    cfg = FrameworkConfig(
        n_features=X.shape[1],
        batch_size=batch_size,
        # Sliding window: 5× batch covers enough samples for stable metrics
        fairness_window_size=batch_size * 5,
        fairness_threshold=threshold,
        # Allow more warmup batches before triggering unlearning.
        # COMPAS has structural bias from batch 1; raising this gives the
        # model time to stabilise before corrections are applied.
        consecutive_violations=7,
        influence_sample_size=200,
        # Smaller top_k keeps candidates ≤ 20 → switches to influence_newton,
        # which makes precise parameter updates rather than the broad
        # gradient_reversal swings that currently over-correct and fail utility.
        top_k_influential=15,
        unlearning_budget=15,
        replay_buffer_size=batch_size * 3,
        # Slightly looser tolerance: gradient_reversal on COMPAS causes small
        # accuracy fluctuations that are acceptable given the fairness gain.
        accuracy_drop_tolerance=0.06,
        ewc_lambda=0.5,
        learning_rate=0.01,   # slightly higher lr: fewer features than Home Credit
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
        dataclasses.replace(cfg)            # ensure fresh config copy
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
    rollback_utility  = sum(1 for a in actions if not a.accepted and "utility" in (a.notes or ""))
    rollback_fairness = sum(1 for a in actions if not a.accepted and "fairness" in (a.notes or ""))
    if rollback_utility or rollback_fairness:
        print(f"  Rollback reasons  : {rollback_utility} utility drop, {rollback_fairness} fairness worsened")
    print()
    print(f"  {'Batch':>6}  {'Method':>20}  {'SPD before':>10}  {'SPD after':>9}  {'Result':>18}")
    for a in actions:
        if a.accepted:
            result = "yes"
        elif "fairness" in (a.notes or ""):
            result = "NO (spd worsened)"
        else:
            result = "NO (utility drop)"
        print(f"  {a.timestamp:>6}  {a.method:>20}  "
              f"{a.fairness_before.spd:>10.4f}  {a.fairness_after.spd:>9.4f}  "
              f"{result:>18}")
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
    axes[0].set_title(
        "Statistical Parity Difference — COMPAS Recidivism (AFU)\n"
        "Protected: Race (Black vs White) | Orange lines = unlearning events"
    )

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
    out = "compas_afu.png"
    plt.savefig(out, dpi=140)
    print(f"Plot saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="AFU on COMPAS Recidivism dataset")
    p.add_argument(
        "--csv",
        default="compas-scores-raw.csv",
        help="Path to compas-scores-raw.csv (default: ./compas-scores-raw.csv)",
    )
    p.add_argument(
        "--batch-size", type=int, default=200,
        help="Samples per streaming batch (default: 200)",
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
        print("Download compas-scores-raw.csv from:")
        print("  https://github.com/propublica/compas-analysis")
        print("  curl -O https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-raw.csv")
        sys.exit(1)
    run(str(csv_path), args.batch_size, args.threshold)
