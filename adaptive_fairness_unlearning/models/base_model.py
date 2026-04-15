"""Step 2 — Base model deployment: online logistic regression."""

from __future__ import annotations

import numpy as np

from ..utils.helpers import sigmoid


class OnlineLogisticRegression:
    """
    Mini-batch online logistic regression with support for influence-function
    computations (gradient and Hessian-vector product).
    """

    def __init__(self, n_features: int, lr: float) -> None:
        self.n_features = n_features
        self.lr = lr
        self.w = np.zeros(n_features)
        self.b = 0.0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1|x) for every row of X."""
        return sigmoid(X @ self.w + self.b)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (threshold 0.5)."""
        return (self.predict(X) >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: float = 1.0,
    ) -> None:
        """
        One mini-batch SGD step.

        Parameters
        ----------
        class_weight : float
            Weight multiplier applied to positive-class (y=1) samples.
            Set to (1 - pos_rate) / pos_rate for balanced training on
            imbalanced datasets (e.g. ~11.0 for 8% default rate).
        """
        n = len(y)
        residual = self.predict(X) - y            # (n,)
        weights = np.where(y == 1, class_weight, 1.0)  # (n,)
        weighted_residual = residual * weights
        self.w -= self.lr * (X.T @ weighted_residual) / n
        self.b -= self.lr * weighted_residual.mean()

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_params(self) -> np.ndarray:
        """Return θ = [w; b] as a flat copy."""
        return np.concatenate([self.w, [self.b]]).copy()

    def set_params(self, theta: np.ndarray) -> None:
        """Restore parameters from a flat vector θ = [w; b]."""
        self.w = theta[:-1].copy()
        self.b = float(theta[-1])

    # ------------------------------------------------------------------
    # Gradient and Hessian
    # ------------------------------------------------------------------

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Per-sample gradients, shape (n_samples, n_features+1).
        grad_w_i = (sigmoid(x_i·w+b) - y_i) * x_i
        grad_b_i = sigmoid(x_i·w+b) - y_i
        """
        residual = (self.predict(X) - y)[:, np.newaxis]  # (n, 1)
        grad_w = residual * X                             # (n, d)
        grad_b = residual                                 # (n, 1)
        return np.hstack([grad_w, grad_b])                # (n, d+1)

    def hessian_vector_product(self, X: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute H·v where H = (1/N) X_aug.T @ diag(p*(1-p)) @ X_aug.
        X_aug is X with a bias column of ones appended.
        """
        n = len(X)
        p = self.predict(X)
        D = p * (1.0 - p)                    # (n,)
        X_aug = np.hstack([X, np.ones((n, 1))])  # (n, d+1)
        # H @ v = (1/n) X_aug.T @ (D * (X_aug @ v))
        Xv = X_aug @ v                        # (n,)
        return (X_aug.T @ (D * Xv)) / n
