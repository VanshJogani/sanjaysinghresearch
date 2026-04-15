"""Step 3 — Online fairness monitoring on a sliding window."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from ..utils.types import FairnessSnapshot
from ..utils.helpers import safe_positive_rate, conditional_positive_rate


class FairnessMonitor:
    """
    Maintains a fixed-size sliding window of (ŷ, y, A) triples and computes
    Statistical Parity Difference (SPD) and Equalized Odds Difference (EOD).
    """

    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._preds: List[int] = []
        self._labels: List[int] = []
        self._groups: List[int] = []

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def update(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        protected: np.ndarray,
    ) -> None:
        """Push a batch into the window; evict oldest entries if overflow."""
        self._preds.extend(y_pred.tolist())
        self._labels.extend(y_true.tolist())
        self._groups.extend(protected.tolist())

        # Trim to window size
        excess = len(self._preds) - self._window_size
        if excess > 0:
            self._preds = self._preds[excess:]
            self._labels = self._labels[excess:]
            self._groups = self._groups[excess:]

    def reset(self) -> None:
        """Clear the window."""
        self._preds = []
        self._labels = []
        self._groups = []

    # ------------------------------------------------------------------
    # Fairness metrics
    # ------------------------------------------------------------------

    def spd(self) -> float:
        """Statistical Parity Difference = |P(ŷ=1|A=0) − P(ŷ=1|A=1)|."""
        if not self._preds:
            return 0.0
        preds = np.array(self._preds)
        groups = np.array(self._groups)

        rate0 = safe_positive_rate(preds[groups == 0])
        rate1 = safe_positive_rate(preds[groups == 1])
        return abs(rate0 - rate1)

    def eod(self) -> float:
        """
        Equalized Odds Difference =
            0.5 * (|TPR_0 - TPR_1| + |FPR_0 - FPR_1|)
        """
        if not self._preds:
            return 0.0
        preds = np.array(self._preds)
        labels = np.array(self._labels)
        groups = np.array(self._groups)

        tpr0 = conditional_positive_rate(preds, labels, groups, group=0, label=1)
        tpr1 = conditional_positive_rate(preds, labels, groups, group=1, label=1)
        fpr0 = conditional_positive_rate(preds, labels, groups, group=0, label=0)
        fpr1 = conditional_positive_rate(preds, labels, groups, group=1, label=0)
        return 0.5 * (abs(tpr0 - tpr1) + abs(fpr0 - fpr1))

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, timestamp: int, accuracy: float) -> FairnessSnapshot:
        """Bundle current metrics into a FairnessSnapshot."""
        return FairnessSnapshot(
            timestamp=timestamp,
            spd=self.spd(),
            eod=self.eod(),
            accuracy=accuracy,
            window_size=len(self._preds),
        )
