"""
compas_bias_demo.py
===================
Demonstrates bias detection and selective unlearning on the ProPublica COMPAS
recidivism dataset with a controlled bias injection.

Dataset: https://github.com/propublica/compas-analysis
Required file: compas-scores-raw.csv  (place in the same directory)

Bias injection strategy
-----------------------
During specified batch windows we flip labels based on the protected attribute:

  Biased against Black defendants:
    - Black (protected=1), y=0 (did NOT reoffend) → flip to y=1 with prob `flip_rate`
      → model sees Black defendants reoffending more often than they actually do
    - White (protected=0), y=1 (DID reoffend)    → flip to y=0 with prob `flip_rate`
      → model sees White defendants reoffending less often than they actually do

  Net effect: model learns to over-predict recidivism for Black defendants and
  under-predict for White defendants.  This surfaces as a rising SPD and EOD.
  The AFU pipeline detects the violation and unlearns the most influential
  biased training samples.

Why label flipping (not feature perturbation)?
    COMPAS is a relatively small dataset (~6K rows after filtering).  An online
    logistic regression on small data can converge to near-constant predictions,
    making feature perturbation invisible in the SPD/EOD metrics.  Label
    flipping forces the model to learn asymmetric positive rates per group,
    which produces a reliable and measurable fairness signal.

Usage
-----
    python compas_bias_demo.py
    python compas_bias_demo.py --csv path/to/compas-scores-raw.csv
    python compas_bias_demo.py --strength 0.50 --start 5 --end 20
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
# 1. Preprocessing (same as compas_demo.py)
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

    # ---- Filter to general recidivism scale only ----------------------------
    df = df[df["DisplayText"] == "Risk of Recidivism"].copy()

    # ---- Filter to Black / White defendants only ----------------------------
    df = df[df["Ethnic_Code_Text"].isin(["African-American", "Caucasian"])].copy()

    y          = (df["DecileScore"] >= 5).astype(int).values
    protected  = (df["Ethnic_Code_Text"] == "African-American").astype(int).values
    time_order = pd.to_datetime(df["Screening_Date"]).astype("int64").values

    drop_cols = {
        "Person_ID", "AssessmentID", "Case_ID",
        "LastName", "FirstName", "MiddleName", "DateOfBirth",
        "Ethnic_Code_Text",
        "Screening_Date",
        "DecileScore", "RawScore", "ScoreText",
        "ScaleSet_ID", "ScaleSet", "DisplayText", "Scale_ID",
        "AssessmentType", "IsCompleted", "IsDeleted",
    }
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    feature_df[num_cols] = feature_df[num_cols].fillna(feature_df[num_cols].median())
    feature_df[cat_cols] = feature_df[cat_cols].fillna("MISSING")
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_df[num_cols] = (
        (feature_df[num_cols] - feature_df[num_cols].mean())
        / (feature_df[num_cols].std() + 1e-8)
    )

    X = feature_df.values.astype(np.float32)

    print(f"  Shape           : {X.shape}")
    print(f"  High-risk rate  : {y.mean():.3f}")
    print(f"  Black proportion: {protected.mean():.3f}")
    return X, y, protected, time_order


# ---------------------------------------------------------------------------
# 2. Bias-injected stream wrapper
# ---------------------------------------------------------------------------

class RaceBiasInjectedStreamLoader:
    """
    Injects racial bias via **label flipping** during specified batch windows.

    Mechanism (biased against Black defendants):
        During [start_batch, end_batch):
          - Black (protected=1), y=0: flip to y=1 with prob `flip_rate`
            → model sees Black defendants scored high-risk more than they actually are
          - White (protected=0), y=1: flip to y=0 with prob `flip_rate`
            → model sees White defendants scored high-risk less than they actually are

    Net effect: SPD and EOD rise as the model learns different recidivism rates
    per racial group.  AFU detects this and unlearns the most influential biased
    samples.

    Swap the group masks to bias in the opposite direction (favouring Black
    defendants / penalising White defendants).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected: np.ndarray,
        time_order: np.ndarray,
        bias_windows: List[Tuple[int, int, float]],
        seed: int = 42,
    ) -> None:
        self._base    = RealWorldStreamLoader(X, y, protected, time_order)
        self._windows = bias_windows    # [(start_batch, end_batch, flip_rate), ...]
        self._rng     = np.random.RandomState(seed)

    def _flip_rate(self, batch_idx: int) -> float:
        for start, end, rate in self._windows:
            if start <= batch_idx < end:
                return rate
        return 0.0

    def stream(self, batch_size: int = 200) -> Generator[DataBatch, None, None]:
        for batch in self._base.stream(batch_size=batch_size):
            rate = self._flip_rate(batch.timestamp)
            if rate > 0.0:
                y_biased = batch.y.copy()

                # Black defendants (protected=1) scored low-risk → flip to high-risk
                black_no_recid = (batch.protected == 1) & (y_biased == 0)
                flip_b = self._rng.rand(black_no_recid.sum()) < rate
                y_biased[np.where(black_no_recid)[0][flip_b]] = 1

                # White defendants (protected=0) scored high-risk → flip to low-risk
                white_recid = (batch.protected == 0) & (y_biased == 1)
                flip_w = self._rng.rand(white_recid.sum()) < rate
                y_biased[np.where(white_recid)[0][flip_w]] = 0

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

    X, y, protected, time_order = load_compas(csv_path)

    bias_windows = [(bias_start, bias_end, bias_strength)]

    print(f"\nBias injection window : batches {bias_start}–{bias_end - 1}, "
          f"flip_rate={bias_strength:.0%}")
    print(f"Effect: {bias_strength:.0%} of Black low-risk scores → flipped to high-risk")
    print(f"        {bias_strength:.0%} of White high-risk scores → flipped to low-risk\n")

    loader = RaceBiasInjectedStreamLoader(
        X, y, protected, time_order, bias_windows
    )

    cfg = FrameworkConfig(
        n_features=X.shape[1],
        batch_size=batch_size,
        fairness_window_size=batch_size * 5,
        fairness_threshold=threshold,
        consecutive_violations=3,
        influence_sample_size=200,
        top_k_influential=20,
        unlearning_budget=30,
        replay_buffer_size=batch_size * 3,
        accuracy_drop_tolerance=0.05,
        ewc_lambda=0.5,
        learning_rate=0.01,
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
    print(f"  {'Batch':>6}  {'SPD':>8}  {'':2}  {'EOD':>8}  {'':2}  {'Accuracy':>9}  {'':>14}")
    print(f"  {'------':>6}  {'---':>8}  {'':2}  {'---':>8}  {'':2}  {'--------':>9}")

    def _in_window(t):
        return any(s <= t < e for s, e, _ in bias_windows)

    step = max(1, len(history) // 30)
    for s in history[::step]:
        flag       = " ← BIAS ACTIVE" if _in_window(s.timestamp) else ""
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
    print(f"  Mean cost    : {np.mean(costs) * 1000:.1f} ms  "
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
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    ts  = [s.timestamp for s in history]
    spd = [s.spd       for s in history]
    eod = [s.eod       for s in history]
    acc = [s.accuracy  for s in history]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Shade bias-injection windows in light red
    for ax in axes:
        for start, end, _ in bias_windows:
            ax.axvspan(start, end - 1, color="red", alpha=0.08, label="_nolegend_")

    axes[0].plot(ts, spd, color="steelblue", linewidth=1.3, label="SPD")
    axes[0].axhline(threshold, color="red", linestyle="--",
                    linewidth=0.9, label=f"threshold τ={threshold}")
    for i, a in enumerate(actions):
        axes[0].axvline(a.timestamp, color="orange", linewidth=1.2, alpha=0.8,
                        label="unlearning" if i == 0 else "_nolegend_")
    axes[0].set_ylabel("SPD")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        "COMPAS Recidivism — Racial Bias Injected (against Black defendants)\n"
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
    out = "compas_bias_afu.png"
    plt.savefig(out, dpi=140)
    print(f"Plot saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="AFU bias detection demo on COMPAS Recidivism dataset"
    )
    p.add_argument("--csv",         default="compas-scores-raw.csv")
    p.add_argument("--batch-size",  type=int,   default=200)
    p.add_argument("--threshold",   type=float, default=0.10)
    p.add_argument("--start",       type=int,   default=5,
                   help="First batch with bias injected (default: 5)")
    p.add_argument("--end",         type=int,   default=20,
                   help="First batch WITHOUT bias (default: 20)")
    p.add_argument("--strength",    type=float, default=0.40,
                   help="Fraction of labels to flip during bias window (default: 0.40)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Download from: https://github.com/propublica/compas-analysis")
        print("  curl -O https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-raw.csv")
        sys.exit(1)
    run(
        str(csv_path),
        batch_size=args.batch_size,
        threshold=args.threshold,
        bias_start=args.start,
        bias_end=args.end,
        bias_strength=args.strength,
    )
