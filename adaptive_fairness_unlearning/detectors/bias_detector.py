"""Step 4 — Bias detection trigger using a consecutive-violation counter."""

from __future__ import annotations

from ..utils.types import FairnessSnapshot


class BiasDetector:
    """
    Raises a fairness-violation flag when SPD or EOD exceeds the threshold
    for *k* consecutive monitoring windows (two-level CUSUM-style check).
    """

    def __init__(self, threshold: float, consecutive_k: int) -> None:
        self._threshold = threshold
        self._consecutive_k = consecutive_k
        self._violation_count: int = 0

    def check(self, snapshot: FairnessSnapshot) -> bool:
        """
        Returns True only when the metric has exceeded the threshold for at
        least consecutive_k windows in a row.
        """
        if snapshot.spd > self._threshold or snapshot.eod > self._threshold:
            self._violation_count += 1
        else:
            self._violation_count = 0
        return self._violation_count >= self._consecutive_k

    def reset(self) -> None:
        """Reset internal state (post-unlearning cooldown)."""
        self._violation_count = 0

    def get_state(self) -> dict:
        """Expose internal state for debugging / audit."""
        return {
            "violation_count": self._violation_count,
            "threshold": self._threshold,
            "consecutive_k": self._consecutive_k,
        }
