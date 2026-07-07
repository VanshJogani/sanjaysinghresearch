"""
run_demo.py — Full demonstration of the Adaptive Fairness Unlearning framework.

Run with:
    python run_demo.py
"""

from __future__ import annotations

import sys

import numpy as np

from adaptive_fairness_unlearning.utils import FrameworkConfig
from adaptive_fairness_unlearning.data.stream import SyntheticStreamGenerator
from adaptive_fairness_unlearning.pipeline import AdaptiveFairUnlearningPipeline
from adaptive_fairness_unlearning.evaluation.benchmarks import Evaluator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DRIFT_SCHEDULE = {0: 0.0, 20: 0.0, 35: 0.785}        # 45° rotation at batch 35
BIAS_WINDOWS = [(25, 40, 3.0)]                          # strong bias injected 25–40


def make_stream(cfg: FrameworkConfig):
    """Return a fresh stream with drift + bias injection."""
    gen = SyntheticStreamGenerator(cfg)
    return gen.stream(
        drift_schedule=DRIFT_SCHEDULE,
        bias_injection_windows=BIAS_WINDOWS,
    )


# ---------------------------------------------------------------------------
# 1. Generate clean pre-training data (no drift, no bias)
# ---------------------------------------------------------------------------

def make_pretrain_data(cfg: FrameworkConfig, n_samples: int = 500):
    """Generate clean historical data for pre-training (no drift, no bias)."""
    pretrain_cfg = FrameworkConfig(
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        batch_size=n_samples,
        n_batches=1,
        noise_std=cfg.noise_std,
        base_protected_correlation=cfg.base_protected_correlation,
        seed=cfg.seed + 1000,  # different seed so data isn't identical to stream
    )
    gen = SyntheticStreamGenerator(pretrain_cfg)
    # Single batch, no drift, no bias injection
    batch = next(gen.stream(drift_schedule=None, bias_injection_windows=None))
    return batch.X, batch.y, batch.protected


# ---------------------------------------------------------------------------
# 2. Run AFU pipeline (with pre-training)
# ---------------------------------------------------------------------------

def run_pipeline(cfg: FrameworkConfig):
    pipe = AdaptiveFairUnlearningPipeline(cfg)

    # Pre-train on clean historical data
    X_pre, y_pre, p_pre = make_pretrain_data(cfg)
    pretrain_history = pipe.fit_initial(X_pre, y_pre, p_pre)

    # Stream the (potentially biased) data
    history, actions = pipe.run(make_stream(cfg))
    return pipe, pretrain_history, history, actions


# ---------------------------------------------------------------------------
# 2. Run comparative evaluation
# ---------------------------------------------------------------------------

def run_comparison(cfg: FrameworkConfig) -> dict:
    ev = Evaluator(cfg)
    return ev.run_all(lambda: make_stream(cfg))


# ---------------------------------------------------------------------------
# 3. Print helpers
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    header = f"{'Method':<22} | {'SPD':>6} | {'EOD':>6} | {'Acc':>6} | {'Violations':>10} | {'Events':>7} | {'Time(s)':>8}"
    print()
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<22} | {r['spd_mean']:>6.4f} | {r['eod_mean']:>6.4f} | "
            f"{r['accuracy_mean']:>6.4f} | {r['n_violations']:>10} | "
            f"{r['n_unlearning_events']:>7} | {r['total_time_seconds']:>8.2f}"
        )
    print()


def print_audit_summary(actions) -> None:
    print("=== Audit Log Summary ===")
    print(f"  Total unlearning events : {len(actions)}")
    accepted = sum(1 for a in actions if a.accepted)
    print(f"  Accepted                : {accepted}")
    print(f"  Rolled back             : {len(actions) - accepted}")
    if actions:
        mean_cost = np.mean([a.cost_seconds for a in actions])
        print(f"  Mean cost (s)           : {mean_cost:.4f}")
        mean_spd_delta = np.mean([a.fairness_before.spd - a.fairness_after.spd for a in actions])
        print(f"  Mean SPD improvement    : {mean_spd_delta:.4f}")
    print()


def print_fairness_trajectory(history, n_show: int = 10) -> None:
    print("=== Fairness Trajectory (AFU, every batch) ===")
    step = max(1, len(history) // n_show)
    print(f"  {'t':>4}  {'SPD':>8}  {'EOD':>8}  {'Acc':>8}")
    for s in history[::step]:
        print(f"  {s.timestamp:>4}  {s.spd:>8.4f}  {s.eod:>8.4f}  {s.accuracy:>8.4f}")
    print()


# ---------------------------------------------------------------------------
# 4. Optional matplotlib plots
# ---------------------------------------------------------------------------

def try_plot(history_dict: dict, cfg: FrameworkConfig) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {"afu": "blue", "periodic": "orange", "fairness_sgd": "green", "static": "red"}

    # Re-run AFU with pre-training for the plot
    cfg2 = FrameworkConfig(
        n_batches=cfg.n_batches,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    pipe = AdaptiveFairUnlearningPipeline(cfg2)
    X_pre, y_pre, p_pre = make_pretrain_data(cfg2)
    pretrain_hist = pipe.fit_initial(X_pre, y_pre, p_pre)
    hist_afu, _ = pipe.run(make_stream(cfg2))

    # Combine pre-train + stream for full timeline
    all_hist = pretrain_hist + hist_afu
    ts = [s.timestamp for s in all_hist]

    # Mark where streaming begins
    stream_start = hist_afu[0].timestamp if hist_afu else 0

    axes[0].plot(ts, [s.spd for s in all_hist], label="AFU", color=colors["afu"])
    axes[0].axhline(cfg.fairness_threshold, color="grey", linestyle="--", label="threshold")
    axes[0].axvline(stream_start, color="black", linestyle=":", alpha=0.5, label="stream start")
    axes[0].set_ylabel("SPD")
    axes[0].legend()
    axes[0].set_title("Statistical Parity Difference over Time")

    axes[1].plot(ts, [s.eod for s in all_hist], label="AFU", color=colors["afu"])
    axes[1].axhline(cfg.fairness_threshold, color="grey", linestyle="--")
    axes[1].axvline(stream_start, color="black", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("EOD")
    axes[1].set_title("Equalized Odds Difference over Time")

    axes[2].plot(ts, [s.accuracy for s in all_hist], label="AFU", color=colors["afu"])
    axes[2].axvline(stream_start, color="black", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xlabel("Batch")
    axes[2].set_title("Predictive Accuracy over Time")

    plt.tight_layout()
    plt.savefig("afu_demo_plot.png", dpi=120)
    print("Plot saved to afu_demo_plot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("   Adaptive Fairness Unlearning — Full Demo")
    print("=" * 60)

    cfg = FrameworkConfig(
        n_batches=50,
        batch_size=200,
        seed=42,
        fairness_threshold=0.05,
        consecutive_violations=2,
        unlearning_budget=30,
    )

    # --- Pipeline run -------------------------------------------------------
    print("\nPre-training on 500 clean samples...")
    pipe, pretrain_history, history, actions = run_pipeline(cfg)

    print(f"  Pre-training converged in {len(pretrain_history)} epoch(s)")
    print(f"  Final pre-train accuracy: {pretrain_history[-1].accuracy:.4f}")
    print(f"  Final pre-train SPD:      {pretrain_history[-1].spd:.4f}")
    print()

    print("Running AFU pipeline on streaming data...")
    print_fairness_trajectory(history, n_show=15)
    print_audit_summary(actions)

    # --- Comparative evaluation ---------------------------------------------
    print("Running comparative evaluation (this may take ~30 s)...")
    results = run_comparison(cfg)
    print_comparison_table(results)

    # --- Plots --------------------------------------------------------------
    try_plot(results, cfg)


if __name__ == "__main__":
    main()
