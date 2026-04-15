"""Step 1 — Data stream simulation and preprocessing."""

from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from ..utils.types import DataBatch, FrameworkConfig


class SyntheticStreamGenerator:
    """
    Generates a binary-classification streaming dataset with controllable
    concept drift and bias injection windows.
    """

    def __init__(self, cfg: FrameworkConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.seed)

        # Base decision-boundary weight (unit-normed, first n_informative dims)
        raw = self._rng.randn(cfg.n_informative)
        self._base_w = raw / (np.linalg.norm(raw) + 1e-10)

        # Orthogonal drift direction
        raw2 = self._rng.randn(cfg.n_informative)
        raw2 -= raw2.dot(self._base_w) * self._base_w
        self._drift_dir = raw2 / (np.linalg.norm(raw2) + 1e-10)

    def _interpolate_angle(
        self,
        batch_idx: int,
        drift_schedule: Optional[Dict[int, float]],
    ) -> float:
        if not drift_schedule:
            return 0.0
        keys = sorted(drift_schedule.keys())
        if batch_idx <= keys[0]:
            return drift_schedule[keys[0]]
        if batch_idx >= keys[-1]:
            return drift_schedule[keys[-1]]
        for i in range(len(keys) - 1):
            lo, hi = keys[i], keys[i + 1]
            if lo <= batch_idx <= hi:
                t = (batch_idx - lo) / (hi - lo)
                return drift_schedule[lo] + t * (drift_schedule[hi] - drift_schedule[lo])
        return 0.0

    def _bias_strength(
        self,
        batch_idx: int,
        bias_injection_windows: Optional[List[Tuple[int, int, float]]],
    ) -> float:
        if not bias_injection_windows:
            return 0.0
        for start, end, strength in bias_injection_windows:
            if start <= batch_idx < end:
                return strength
        return 0.0

    def stream(
        self,
        drift_schedule: Optional[Dict[int, float]] = None,
        bias_injection_windows: Optional[List[Tuple[int, int, float]]] = None,
    ) -> Generator[DataBatch, None, None]:
        cfg = self.cfg
        global_idx = 0
        for t in range(cfg.n_batches):
            # Rotated weight vector
            angle = self._interpolate_angle(t, drift_schedule)
            w = np.cos(angle) * self._base_w + np.sin(angle) * self._drift_dir

            # Feature matrix
            X = self._rng.randn(cfg.batch_size, cfg.n_features)

            # Protected attribute ~ Bernoulli(0.5)
            protected = (self._rng.rand(cfg.batch_size) > 0.5).astype(int)

            # Logits
            bias_inj = self._bias_strength(t, bias_injection_windows)
            logit = (
                X[:, : cfg.n_informative] @ w
                + cfg.base_protected_correlation * (2 * protected - 1)
                + bias_inj * (2 * protected - 1)
                + self._rng.randn(cfg.batch_size) * cfg.noise_std
            )

            # Labels via Bernoulli(sigmoid(logit))
            probs = 1.0 / (1.0 + np.exp(-logit))
            y = (self._rng.rand(cfg.batch_size) < probs).astype(int)

            indices = np.arange(global_idx, global_idx + cfg.batch_size)
            global_idx += cfg.batch_size

            yield DataBatch(
                X=X,
                y=y,
                protected=protected,
                timestamp=t,
                indices=indices,
            )


class RealWorldStreamLoader:
    """
    Wraps any pre-loaded tabular dataset.  Sorts by time_order and chunks
    into sequential DataBatch objects.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected: np.ndarray,
        time_order: np.ndarray,
    ) -> None:
        order = np.argsort(time_order)
        self.X = X[order]
        self.y = y[order]
        self.protected = protected[order]

    def stream(self, batch_size: int = 500) -> Generator[DataBatch, None, None]:
        n = len(self.y)
        global_idx = 0
        t = 0
        while global_idx < n:
            end = min(global_idx + batch_size, n)
            yield DataBatch(
                X=self.X[global_idx:end],
                y=self.y[global_idx:end],
                protected=self.protected[global_idx:end],
                timestamp=t,
                indices=np.arange(global_idx, end),
            )
            global_idx = end
            t += 1
