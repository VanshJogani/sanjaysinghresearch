"""Step 9 — Evaluation protocol with three baselines."""

from __future__ import annotations

import time
from typing import Callable, Dict, Generator, List

import numpy as np

from ..utils.types import DataBatch, FairnessSnapshot, FrameworkConfig
from ..models.base_model import OnlineLogisticRegression
from ..monitors.fairness_monitor import FairnessMonitor
from ..pipeline import AdaptiveFairUnlearningPipeline


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _count_violations(history: List[FairnessSnapshot], threshold: float) -> int:
    return sum(1 for s in history if s.spd > threshold or s.eod > threshold)


def _summarise(
    history: List[FairnessSnapshot],
    threshold: float,
    elapsed: float,
    n_unlearning: int = 0,
) -> dict:
    spd = np.array([s.spd for s in history])
    eod = np.array([s.eod for s in history])
    acc = np.array([s.accuracy for s in history])
    return {
        "spd_mean": float(spd.mean()) if len(spd) else 0.0,
        "spd_std": float(spd.std()) if len(spd) else 0.0,
        "spd_max": float(spd.max()) if len(spd) else 0.0,
        "eod_mean": float(eod.mean()) if len(eod) else 0.0,
        "eod_std": float(eod.std()) if len(eod) else 0.0,
        "eod_max": float(eod.max()) if len(eod) else 0.0,
        "accuracy_mean": float(acc.mean()) if len(acc) else 0.0,
        "n_violations": _count_violations(history, threshold),
        "total_time_seconds": elapsed,
        "n_unlearning_events": n_unlearning,
    }


# ---------------------------------------------------------------------------
# Baseline 1 — Periodic full retraining
# ---------------------------------------------------------------------------

class PeriodicRetrainer:
    """Retrains the model from scratch every `retrain_every` samples."""

    def __init__(self, cfg: FrameworkConfig, retrain_every: int = 1000) -> None:
        self._cfg = cfg
        self._retrain_every = retrain_every

    def run(self, stream: Generator[DataBatch, None, None]) -> List[FairnessSnapshot]:
        model = OnlineLogisticRegression(self._cfg.n_features, self._cfg.learning_rate)
        monitor = FairnessMonitor(self._cfg.fairness_window_size)
        history: List[FairnessSnapshot] = []

        acc_X: List[np.ndarray] = []
        acc_y: List[np.ndarray] = []
        samples_since_retrain = 0

        for batch in stream:
            acc_X.append(batch.X)
            acc_y.append(batch.y)
            samples_since_retrain += len(batch.y)

            if samples_since_retrain >= self._retrain_every:
                X_all = np.vstack(acc_X)
                y_all = np.concatenate(acc_y)
                model = OnlineLogisticRegression(
                    self._cfg.n_features, self._cfg.learning_rate
                )
                # Mini-batch retrain for a few passes
                for _ in range(3):
                    model.update(X_all, y_all)
                samples_since_retrain = 0
            else:
                model.update(batch.X, batch.y)

            y_pred = model.predict_labels(batch.X)
            acc = float((y_pred == batch.y).mean())
            monitor.update(y_pred, batch.y, batch.protected)
            history.append(monitor.snapshot(batch.timestamp, acc))

        return history


# ---------------------------------------------------------------------------
# Baseline 2 — Fairness-regularised online SGD
# ---------------------------------------------------------------------------

class FairnessRegularizedSGD:
    """
    Online SGD with a fairness penalty added at each step:
        L = L_task + lambda_fair * (SPD² + EOD²)
    """

    def __init__(self, cfg: FrameworkConfig, lambda_fair: float = 0.1) -> None:
        self._cfg = cfg
        self._lambda_fair = lambda_fair

    def run(self, stream: Generator[DataBatch, None, None]) -> List[FairnessSnapshot]:
        model = OnlineLogisticRegression(self._cfg.n_features, self._cfg.learning_rate)
        monitor = FairnessMonitor(self._cfg.fairness_window_size)
        history: List[FairnessSnapshot] = []

        for batch in stream:
            y_pred = model.predict_labels(batch.X)
            acc = float((y_pred == batch.y).mean())
            monitor.update(y_pred, batch.y, batch.protected)
            snap = monitor.snapshot(batch.timestamp, acc)

            # Task gradient step
            model.update(batch.X, batch.y)

            # Fairness penalty: nudge weights to reduce SPD/EOD
            lf = self._lambda_fair
            spd_val = snap.spd
            eod_val = snap.eod
            if spd_val + eod_val > 0:
                g0 = batch.X[batch.protected == 0]
                g1 = batch.X[batch.protected == 1]
                if len(g0) > 0 and len(g1) > 0:
                    mean_diff = g0.mean(axis=0) - g1.mean(axis=0)
                    penalty = np.concatenate(
                        [lf * (spd_val + eod_val) * mean_diff, [0.0]]
                    )
                    theta = model.get_params()
                    theta -= model.lr * penalty
                    model.set_params(theta)

            history.append(snap)

        return history


# ---------------------------------------------------------------------------
# Baseline 3 — Static unlearning (one-shot, no adaptation)
# ---------------------------------------------------------------------------

class StaticUnlearner:
    """
    Runs one-shot unlearning on the first batch, then does plain online SGD.
    """

    def __init__(self, cfg: FrameworkConfig) -> None:
        self._cfg = cfg

    def run(self, stream: Generator[DataBatch, None, None]) -> List[FairnessSnapshot]:
        model = OnlineLogisticRegression(self._cfg.n_features, self._cfg.learning_rate)
        monitor = FairnessMonitor(self._cfg.fairness_window_size)
        history: List[FairnessSnapshot] = []
        first = True

        for batch in stream:
            if first:
                # Train on first batch then do a gradient-reversal unlearning
                model.update(batch.X, batch.y)
                # Reverse on high-protected group (rough proxy)
                mask = batch.protected == 1
                if mask.sum() > 0:
                    for _ in range(3):
                        theta = model.get_params()
                        g = model.gradient(batch.X[mask], batch.y[mask]).mean(axis=0)
                        theta += model.lr * g
                        model.set_params(theta)
                first = False
            else:
                model.update(batch.X, batch.y)

            y_pred = model.predict_labels(batch.X)
            acc = float((y_pred == batch.y).mean())
            monitor.update(y_pred, batch.y, batch.protected)
            history.append(monitor.snapshot(batch.timestamp, acc))

        return history


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs the AFU pipeline and all three baselines on the same stream,
    collecting comparable metrics.
    """

    def __init__(self, cfg: FrameworkConfig) -> None:
        self._cfg = cfg

    def run_all(self, stream_factory: Callable) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        threshold = self._cfg.fairness_threshold

        # --- AFU pipeline ---
        t0 = time.time()
        from ..utils.types import FrameworkConfig as _FC
        import dataclasses
        cfg_copy = dataclasses.replace(self._cfg)
        pipe = AdaptiveFairUnlearningPipeline(cfg_copy)
        history_afu, actions_afu = pipe.run(stream_factory())
        results["afu"] = _summarise(
            history_afu, threshold, time.time() - t0, len(actions_afu)
        )

        # --- Periodic retraining ---
        t0 = time.time()
        retrain = PeriodicRetrainer(self._cfg, retrain_every=500)
        results["periodic"] = _summarise(
            retrain.run(stream_factory()), threshold, time.time() - t0
        )

        # --- Fairness-regularised SGD ---
        t0 = time.time()
        fair_sgd = FairnessRegularizedSGD(self._cfg, lambda_fair=0.1)
        results["fairness_sgd"] = _summarise(
            fair_sgd.run(stream_factory()), threshold, time.time() - t0
        )

        # --- Static unlearning ---
        t0 = time.time()
        static = StaticUnlearner(self._cfg)
        results["static"] = _summarise(
            static.run(stream_factory()), threshold, time.time() - t0
        )

        return results
