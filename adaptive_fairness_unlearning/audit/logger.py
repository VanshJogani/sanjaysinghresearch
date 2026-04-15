"""Step 8 — Audit trail and adaptive threshold control."""

from __future__ import annotations

from typing import List

from ..utils.types import FairnessSnapshot, UnlearningAction


class AuditLogger:
    """
    Records every unlearning event and provides summary statistics plus an
    adaptive fairness-threshold feedback loop.
    """

    def __init__(self) -> None:
        self._history: List[UnlearningAction] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, action: UnlearningAction) -> None:
        self._history.append(action)

    def get_history(self) -> List[UnlearningAction]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary_stats(self) -> dict:
        total = len(self._history)
        if total == 0:
            return {
                "total_events": 0,
                "accepted": 0,
                "rejected": 0,
                "mean_cost_seconds": 0.0,
                "mean_spd_improvement": 0.0,
                "mean_eod_improvement": 0.0,
                "violation_frequency": 0.0,
            }

        accepted = sum(1 for a in self._history if a.accepted)
        mean_cost = sum(a.cost_seconds for a in self._history) / total
        spd_impr = sum(
            a.fairness_before.spd - a.fairness_after.spd for a in self._history
        ) / total
        eod_impr = sum(
            a.fairness_before.eod - a.fairness_after.eod for a in self._history
        ) / total

        return {
            "total_events": total,
            "accepted": accepted,
            "rejected": total - accepted,
            "mean_cost_seconds": mean_cost,
            "mean_spd_improvement": spd_impr,
            "mean_eod_improvement": eod_impr,
            "violation_frequency": total,
        }

    # ------------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------------

    def adaptive_threshold(self, current_tau: float, lookback: int = 10) -> float:
        """
        Tighten τ by 10 % if > 50 % of the last lookback events were
        violations (accepted=False).  Loosen by 5 % if < 20 %.
        Clamped to [0.03, 0.20].
        """
        recent = self._history[-lookback:]
        if not recent:
            return current_tau

        violation_rate = sum(1 for a in recent if not a.accepted) / len(recent)

        if violation_rate > 0.5:
            current_tau *= 0.9
        elif violation_rate < 0.2:
            current_tau *= 1.05

        return float(max(0.03, min(0.20, current_tau)))

    # ------------------------------------------------------------------
    # Tabular view
    # ------------------------------------------------------------------

    def to_dataframe(self) -> dict:
        """Return a dict-of-lists suitable for pd.DataFrame or direct use."""
        rows = {
            "timestamp": [],
            "method": [],
            "n_candidates": [],
            "spd_before": [],
            "spd_after": [],
            "eod_before": [],
            "eod_after": [],
            "utility_before": [],
            "utility_after": [],
            "cost_seconds": [],
            "accepted": [],
            "notes": [],
        }
        for a in self._history:
            rows["timestamp"].append(a.timestamp)
            rows["method"].append(a.method)
            rows["n_candidates"].append(len(a.candidates))
            rows["spd_before"].append(a.fairness_before.spd)
            rows["spd_after"].append(a.fairness_after.spd)
            rows["eod_before"].append(a.fairness_before.eod)
            rows["eod_after"].append(a.fairness_after.eod)
            rows["utility_before"].append(a.utility_before)
            rows["utility_after"].append(a.utility_after)
            rows["cost_seconds"].append(a.cost_seconds)
            rows["accepted"].append(a.accepted)
            rows["notes"].append(a.notes)
        return rows

    def explainability_report(self, action: UnlearningAction) -> dict:
        """Top-k influence scores and delta-fairness for a given event."""
        candidates_info = sorted(
            [{"index": c.index, "influence": c.influence_score} for c in action.candidates],
            key=lambda x: x["influence"],
            reverse=True,
        )[:10]
        return {
            "timestamp": action.timestamp,
            "method": action.method,
            "top_candidates": candidates_info,
            "spd_delta": action.fairness_before.spd - action.fairness_after.spd,
            "eod_delta": action.fairness_before.eod - action.fairness_after.eod,
            "utility_delta": action.utility_after - action.utility_before,
        }
