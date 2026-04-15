"""Pure utility functions — no state, no side effects."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def safe_positive_rate(preds: np.ndarray) -> float:
    """P(ŷ=1); returns 0.0 if the array is empty."""
    if len(preds) == 0:
        return 0.0
    return float(np.mean(preds))


def conditional_positive_rate(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    group: int,
    label: int,
) -> float:
    """P(ŷ=1 | Y=label, A=group). Returns 0.0 if the subset is empty."""
    mask = (groups == group) & (labels == label)
    subset = preds[mask]
    if len(subset) == 0:
        return 0.0
    return float(np.mean(subset))
