"""Step 5 — Bias source identification via influence functions (LiSSA)."""

from __future__ import annotations

from typing import List

import numpy as np

from ..utils.types import FrameworkConfig, UnlearningCandidate
from ..utils.helpers import conditional_positive_rate


class InfluenceEstimator:
    """
    Estimates the influence of each training point on the current fairness
    violation using influence functions with the LiSSA approximation for
    the Hessian inverse.
    """

    def __init__(self, model, cfg: FrameworkConfig) -> None:
        self._model = model
        self._cfg = cfg
        self._rng = np.random.RandomState(cfg.seed)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _fairness_gradient(
        self,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        prot_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Approximate ∇θ L_fair as the difference of gradients between groups.
        L_fair ≈ (positive_rate_group0 - positive_rate_group1)^2
        Gradient direction: grad on group-0 minus grad on group-1 eval points.
        """
        g0 = X_eval[prot_eval == 0]
        y0 = y_eval[prot_eval == 0]
        g1 = X_eval[prot_eval == 1]
        y1 = y_eval[prot_eval == 1]

        d = self._model.get_params().shape[0]
        grad_fair = np.zeros(d)

        if len(g0) > 0:
            grad_fair += self._model.gradient(g0, y0).mean(axis=0)
        if len(g1) > 0:
            grad_fair -= self._model.gradient(g1, y1).mean(axis=0)
        return grad_fair

    def _lissa(
        self,
        v: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 10,
        damping: float = 0.01,
    ) -> np.ndarray:
        """
        LiSSA: approximate H⁻¹ · v by iterating
            v_{t+1} = fairness_grad + (I - H) · v_t
        with damping (H + λI) for numerical stability.
        """
        est = v.copy()
        n = len(y_train)
        sample_size = min(self._cfg.influence_sample_size, n)

        for _ in range(n_iter):
            idx = self._rng.choice(n, sample_size, replace=False)
            Xs, ys = X_train[idx], y_train[idx]
            hvp = self._model.hessian_vector_product(Xs, est)
            hvp += damping * est  # damping
            est = v + est - hvp

        return est

    def compute_influences(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        prot_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        prot_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Compute influence score for each training point on the fairness loss.

        I_F(z_i) ≈ −v^T · ∇θ L(z_i, θ)
        where v = H⁻¹ · ∇θ L_fair

        Returns array of shape (n_train,).
        """
        if len(X_train) == 0:
            return np.zeros(0)

        fair_grad = self._fairness_gradient(X_eval, y_eval, prot_eval)
        v = self._lissa(fair_grad, X_train, y_train)

        # Per-sample training gradients: (n_train, d+1)
        grads = self._model.gradient(X_train, y_train)
        scores = -(grads @ v)  # (n_train,)
        return scores

    def get_top_k(
        self,
        scores: np.ndarray,
        k: int,
        batch_timestamps: np.ndarray,
    ) -> List[UnlearningCandidate]:
        """Return the k points with the highest positive influence scores."""
        if len(scores) == 0:
            return []
        k = min(k, len(scores))
        top_idx = np.argsort(scores)[-k:][::-1]
        return [
            UnlearningCandidate(
                index=int(i),
                influence_score=float(scores[i]),
                batch_timestamp=int(batch_timestamps[i]),
            )
            for i in top_idx
            if scores[i] > 0
        ]
