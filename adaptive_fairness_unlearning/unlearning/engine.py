"""Step 6 — Selective unlearning engine (three mechanisms + auto-dispatch)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..utils.types import FrameworkConfig, UnlearningCandidate


class SelectiveUnlearner:
    """
    Three unlearning mechanisms dispatched by candidate-set size:
      - influence_newton  : |U| ≤ 20
      - gradient_reversal : 20 < |U| ≤ 100
      - reweight          : |U| > 100
    """

    def __init__(self, model, cfg: FrameworkConfig) -> None:
        self._model = model
        self._cfg = cfg
        self._rng = np.random.RandomState(cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def auto_select_method(self, n_candidates: int) -> str:
        if n_candidates <= 20:
            return "influence_newton"
        if n_candidates <= 100:
            return "gradient_reversal"
        return "reweight"

    def unlearn(
        self,
        candidates: List[UnlearningCandidate],
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str,
        replay_X: Optional[np.ndarray] = None,
        replay_y: Optional[np.ndarray] = None,
        max_step_norm: float = 0.10,
    ) -> np.ndarray:
        """Dispatch to the chosen mechanism; return updated parameter vector."""
        if not candidates:
            return self._model.get_params()

        idx = np.array([c.index for c in candidates])
        # Clamp indices to valid range
        idx = idx[idx < len(X_train)]
        if len(idx) == 0:
            return self._model.get_params()

        X_cand = X_train[idx]
        y_cand = y_train[idx]

        if method == "influence_newton":
            # Pass the full training set for HVP (Hessian reference) while
            # X_cand/y_cand are only used for the gradient sum.
            return self._influence_newton(
                X_cand, y_cand, X_train, y_train, max_step_norm=max_step_norm
            )
        if method == "gradient_reversal":
            return self._gradient_reversal(X_cand, y_cand, replay_X, replay_y)
        return self._reweight(X_cand, y_cand, replay_X, replay_y)

    # ------------------------------------------------------------------
    # Mechanism A — Newton step via influence functions
    # ------------------------------------------------------------------

    def _influence_newton(
        self,
        X_cand: np.ndarray,
        y_cand: np.ndarray,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        damping: float = 0.01,
        max_step_norm: float = 0.10,
    ) -> np.ndarray:
        """θ_new = θ_old + clip(H⁻¹ · Σ ∇θ L(z, θ_old), max_step_norm)"""
        theta = self._model.get_params()
        # Gradient sum over the candidates only (points to be unlearned)
        grad_sum = self._model.gradient(X_cand, y_cand).sum(axis=0)  # (d+1,)

        # Approximate H⁻¹ · grad_sum via LiSSA: v_{t+1} = grad_sum + (I - H_d)·v_t
        # where H_d = H + damping·I.  Critically, the HVP must use the full
        # training reference set (X_ref), NOT just the candidates.  Using only
        # 15 candidate samples to estimate H produces a highly noisy direction
        # that randomly helps or hurts fairness.  X_ref is the replay buffer
        # (~600 samples) which gives a stable, representative Hessian.
        scale = max(float(np.linalg.norm(grad_sum)), 1e-8)
        v = grad_sum / scale
        g_norm = v.copy()
        for _ in range(10):
            hvp = self._model.hessian_vector_product(X_ref, v)  # full dataset HVP
            hvp += damping * v          # (H + λI)·v
            v = g_norm + v - hvp        # v_{t+1} = g + (I - H_d)·v_t
        v *= scale                      # undo normalisation

        # Clip step to max_step_norm (adaptive: large when SPD is far from
        # threshold, small when close — prevents both inert steps and accuracy
        # collapse).
        step = v
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step_norm:
            step *= max_step_norm / step_norm

        theta_new = theta + step
        self._model.set_params(theta_new)
        return theta_new

    # ------------------------------------------------------------------
    # Mechanism B — Gradient reversal / anti-fine-tuning
    # ------------------------------------------------------------------

    def _gradient_reversal(
        self,
        X_cand: np.ndarray,
        y_cand: np.ndarray,
        replay_X: Optional[np.ndarray],
        replay_y: Optional[np.ndarray],
        steps: int = 5,
    ) -> np.ndarray:
        lr = self._cfg.learning_rate
        for _ in range(steps):
            theta = self._model.get_params()
            # Gradient ASCENT on harmful points (forget)
            grad_forget = self._model.gradient(X_cand, y_cand).mean(axis=0)
            theta += lr * grad_forget
            # Gradient DESCENT on replay buffer (retain)
            if replay_X is not None and len(replay_X) > 0:
                grad_retain = self._model.gradient(replay_X, replay_y).mean(axis=0)
                theta -= lr * grad_retain
            self._model.set_params(theta)
        return self._model.get_params()

    # ------------------------------------------------------------------
    # Mechanism C — Reweight (zero out harmful, re-optimise on replay)
    # ------------------------------------------------------------------

    def _reweight(
        self,
        X_cand: np.ndarray,
        y_cand: np.ndarray,
        replay_X: Optional[np.ndarray],
        replay_y: Optional[np.ndarray],
        steps: int = 10,
    ) -> np.ndarray:
        # Re-optimise exclusively on the replay buffer (harmful points zeroed)
        if replay_X is not None and len(replay_X) > 0:
            lr = self._cfg.learning_rate
            for _ in range(steps):
                theta = self._model.get_params()
                grad = self._model.gradient(replay_X, replay_y).mean(axis=0)
                theta -= lr * grad
                self._model.set_params(theta)
        return self._model.get_params()
