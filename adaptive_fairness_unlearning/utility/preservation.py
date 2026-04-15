"""Step 7 — Utility preservation: replay buffer and EWC-based recovery."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..utils.types import DataBatch, FrameworkConfig


class ReplayBuffer:
    """
    Reservoir of recent, fairly-behaved data points used for utility
    preservation after unlearning.
    """

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._protected: Optional[np.ndarray] = None
        self._rng = np.random.RandomState(0)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, batch: DataBatch) -> None:
        """Append data; trim to max_size by discarding oldest samples."""
        if self._X is None:
            self._X = batch.X.copy()
            self._y = batch.y.copy()
            self._protected = batch.protected.copy()
        else:
            self._X = np.vstack([self._X, batch.X])
            self._y = np.concatenate([self._y, batch.y])
            self._protected = np.concatenate([self._protected, batch.protected])

        if len(self._y) > self._max_size:
            self._X = self._X[-self._max_size:]
            self._y = self._y[-self._max_size:]
            self._protected = self._protected[-self._max_size:]

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return 0 if self._y is None else len(self._y)

    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._X is None:
            return np.zeros((0, 1)), np.zeros(0, int), np.zeros(0, int)
        return self._X.copy(), self._y.copy(), self._protected.copy()

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._X is None or len(self._y) == 0:
            return np.zeros((0, 1)), np.zeros(0, int), np.zeros(0, int)
        n = min(n, len(self._y))
        idx = self._rng.choice(len(self._y), n, replace=False)
        return self._X[idx], self._y[idx], self._protected[idx]


class UtilityPreserver:
    """
    Preserves model utility after selective unlearning via fine-tuning,
    Elastic Weight Consolidation (EWC), and accuracy-triggered recovery.
    """

    def __init__(self, model, cfg: FrameworkConfig) -> None:
        self._model = model
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(self, replay_buffer: ReplayBuffer, steps: int = 10) -> None:
        """Run SGD steps on replay buffer samples."""
        if replay_buffer.size == 0:
            return
        X, y, _ = replay_buffer.get_all()
        lr = self._cfg.learning_rate
        for _ in range(steps):
            self._model.update(X, y)

    # ------------------------------------------------------------------
    # Fisher / EWC
    # ------------------------------------------------------------------

    def compute_fisher(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Diagonal Fisher information matrix ≈ mean of squared per-sample
        gradients, shape (d+1,).
        """
        if len(X) == 0:
            return np.ones(self._model.get_params().shape[0])
        grads = self._model.gradient(X, y)  # (n, d+1)
        return np.mean(grads ** 2, axis=0)

    def ewc_regularize(
        self,
        old_params: np.ndarray,
        fisher: np.ndarray,
        steps: int = 10,
        replay_buffer: Optional[ReplayBuffer] = None,
    ) -> None:
        """
        Fine-tune with EWC penalty:
            L_total = L_task(replay) + ewc_lambda * Σ F_k (θ_k - θ*_k)²
        """
        if replay_buffer is None or replay_buffer.size == 0:
            return
        X, y, _ = replay_buffer.get_all()
        lr = self._cfg.learning_rate
        lam = self._cfg.ewc_lambda

        for _ in range(steps):
            theta = self._model.get_params()
            # Task gradient
            grads_task = self._model.gradient(X, y).mean(axis=0)
            # EWC penalty gradient: 2 * ewc_lambda * F * (θ - θ*)
            ewc_grad = 2.0 * lam * fisher * (theta - old_params)
            theta -= lr * (grads_task + ewc_grad)
            self._model.set_params(theta)

    # ------------------------------------------------------------------
    # Recovery check
    # ------------------------------------------------------------------

    def check_and_recover(
        self,
        val_X: np.ndarray,
        val_y: np.ndarray,
        old_accuracy: float,
        replay_buffer: ReplayBuffer,
    ) -> bool:
        """
        Evaluate accuracy; if drop > tolerance, try EWC recovery.
        Returns True if accuracy is acceptable (or recovered), False otherwise.
        """
        if len(val_X) == 0:
            return True

        preds = self._model.predict_labels(val_X)
        new_acc = float((preds == val_y).mean())

        if (old_accuracy - new_acc) <= self._cfg.accuracy_drop_tolerance:
            return True

        # Attempt recovery
        old_params = self._model.get_params()
        fisher = self.compute_fisher(val_X, val_y)
        self.ewc_regularize(old_params, fisher, steps=25, replay_buffer=replay_buffer)

        # Re-evaluate
        preds2 = self._model.predict_labels(val_X)
        recovered_acc = float((preds2 == val_y).mean())
        return (old_accuracy - recovered_acc) <= self._cfg.accuracy_drop_tolerance
