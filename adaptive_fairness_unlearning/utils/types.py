"""Shared data structures for the AFU framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DataBatch:
    """One chunk from the data stream."""
    X: np.ndarray
    y: np.ndarray
    protected: np.ndarray
    timestamp: int
    indices: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FairnessSnapshot:
    """Fairness and utility metrics at a point in time."""
    timestamp: int
    spd: float
    eod: float
    accuracy: float
    window_size: int
    extra: dict = field(default_factory=dict)


@dataclass
class UnlearningCandidate:
    """A data point flagged as a bias source."""
    index: int
    influence_score: float
    batch_timestamp: int
    features: Optional[np.ndarray] = None


@dataclass
class UnlearningAction:
    """Full record of one unlearning event."""
    timestamp: int
    candidates: List[UnlearningCandidate]
    fairness_before: FairnessSnapshot
    fairness_after: FairnessSnapshot
    utility_before: float
    utility_after: float
    method: str
    cost_seconds: float
    accepted: bool
    notes: str = ""


@dataclass
class FrameworkConfig:
    """Single source of truth for all hyperparameters."""
    # Data generation
    n_features: int = 10
    n_informative: int = 5
    batch_size: int = 200
    n_batches: int = 50
    noise_std: float = 0.3
    base_protected_correlation: float = 0.05
    seed: int = 42

    # Fairness monitoring
    fairness_window_size: int = 1000
    fairness_threshold: float = 0.10
    consecutive_violations: int = 3

    # Attribution
    influence_sample_size: int = 200
    top_k_influential: int = 20

    # Unlearning
    unlearning_budget: int = 30
    replay_buffer_size: int = 500

    # Utility preservation
    accuracy_drop_tolerance: float = 0.05
    ewc_lambda: float = 0.5

    # Optimisation
    learning_rate: float = 0.01
