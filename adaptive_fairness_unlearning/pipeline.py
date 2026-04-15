"""Step 10 — Main loop: AdaptiveFairUnlearningPipeline (Algorithm 1)."""

from __future__ import annotations

import time
from typing import Generator, List, Tuple

import numpy as np

from .utils.types import (
    DataBatch,
    FairnessSnapshot,
    FrameworkConfig,
    UnlearningAction,
)
from .models.base_model import OnlineLogisticRegression
from .monitors.fairness_monitor import FairnessMonitor
from .detectors.bias_detector import BiasDetector
from .attribution.influence import InfluenceEstimator
from .unlearning.engine import SelectiveUnlearner
from .utility.preservation import ReplayBuffer, UtilityPreserver
from .audit.logger import AuditLogger


class AdaptiveFairUnlearningPipeline:
    """
    Wires all modules together into the streaming main loop described in
    Algorithm 1 of the implementation plan.
    """

    def __init__(self, cfg: FrameworkConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.seed)
        self.model = OnlineLogisticRegression(cfg.n_features, cfg.learning_rate)
        self.monitor = FairnessMonitor(cfg.fairness_window_size)
        self.detector = BiasDetector(cfg.fairness_threshold, cfg.consecutive_violations)
        self.influence_estimator = InfluenceEstimator(self.model, cfg)
        self.unlearner = SelectiveUnlearner(self.model, cfg)
        self.replay_buffer = ReplayBuffer(cfg.replay_buffer_size)
        self.preserver = UtilityPreserver(self.model, cfg)
        self.audit = AuditLogger()

    def run(
        self,
        stream: Generator[DataBatch, None, None],
    ) -> Tuple[List[FairnessSnapshot], List[UnlearningAction]]:
        """
        Process every batch in the stream.

        Returns
        -------
        history : list of FairnessSnapshot (one per batch)
        audit   : list of UnlearningAction (one per unlearning event)
        """
        history: List[FairnessSnapshot] = []

        for batch in stream:
            # ---- Predict & monitor ----------------------------------------
            y_pred = self.model.predict_labels(batch.X)
            accuracy = float((y_pred == batch.y).mean())

            self.monitor.update(y_pred, batch.y, batch.protected)
            snapshot = self.monitor.snapshot(batch.timestamp, accuracy)
            history.append(snapshot)

            # ---- Check for fairness violation --------------------------------
            if self.detector.check(snapshot):
                t0 = time.time()

                # Attribution: use replay buffer as training data proxy
                buf_X, buf_y, buf_p = self.replay_buffer.get_all()

                if len(buf_X) > 0:
                    scores = self.influence_estimator.compute_influences(
                        buf_X, buf_y, buf_p,
                        batch.X, batch.y, batch.protected,
                    )
                    batch_ts = np.zeros(len(buf_y), dtype=int)
                    candidates = self.influence_estimator.get_top_k(
                        scores, self.cfg.unlearning_budget, batch_ts
                    )
                else:
                    candidates = []

                # Save state for possible rollback
                old_params = self.model.get_params()

                # Use replay buffer as the utility reference.  The current batch
                # is unseen data the model has never trained on; measuring its
                # pre-training accuracy and then comparing to post-unlearning
                # accuracy on the same unseen batch always shows a drop (the
                # model hasn't been updated yet in either case), causing every
                # unlearning event to be rejected.  Replay buffer samples are
                # data the model has already learned — a fair before/after check.
                val_n = min(200, self.replay_buffer.size)
                val_X_ref, val_y_ref, _ = self.replay_buffer.sample(val_n)
                if len(val_X_ref) > 0:
                    old_acc = float(
                        (self.model.predict_labels(val_X_ref) == val_y_ref).mean()
                    )
                else:
                    val_X_ref, val_y_ref = batch.X, batch.y
                    old_acc = accuracy

                # Unlearn
                if candidates:
                    method = self.unlearner.auto_select_method(len(candidates))
                    # Build a retain set that excludes the candidate indices so
                    # gradient reversal / reweight don't cancel themselves out.
                    cand_idx = set(c.index for c in candidates)
                    all_idx = np.arange(len(buf_y))
                    retain_mask = np.array([i not in cand_idx for i in all_idx])
                    if retain_mask.any():
                        retain_pool_X = buf_X[retain_mask]
                        retain_pool_y = buf_y[retain_mask]
                        rep_n = min(100, len(retain_pool_y))
                        rng_idx = self._rng.choice(len(retain_pool_y), rep_n, replace=False)
                        rep_X, rep_y = retain_pool_X[rng_idx], retain_pool_y[rng_idx]
                    else:
                        rep_n = min(100, self.replay_buffer.size)
                        rep_X, rep_y, _ = self.replay_buffer.sample(rep_n)
                    # Adaptive step size: large when SPD is far above threshold
                    # (aggressive correction is worth the accuracy cost), small
                    # when SPD is near threshold (gentle nudge, preserve accuracy).
                    spd_excess = max(0.0, snapshot.spd - self.cfg.fairness_threshold)
                    max_step_norm = float(np.clip(0.5 * spd_excess, 0.02, 0.10))
                    self.unlearner.unlearn(
                        candidates, buf_X, buf_y, method, rep_X, rep_y,
                        max_step_norm=max_step_norm,
                    )
                else:
                    method = "none"

                # Utility check + EWC recovery
                recovered = self.preserver.check_and_recover(
                    val_X_ref, val_y_ref, old_acc, self.replay_buffer
                )

                cost = time.time() - t0

                # Re-evaluate fairness on the rolling window so SPD is
                # computed over the same 1000-sample history as `snapshot`,
                # not over a single noisy 200-sample batch.
                new_preds_batch = self.model.predict_labels(batch.X)
                temp_monitor = FairnessMonitor(self.cfg.fairness_window_size)
                # Replay the existing window history through the new model
                for p, lbl, grp in zip(
                    self.monitor._preds, self.monitor._labels, self.monitor._groups
                ):
                    temp_monitor._preds.append(p)
                    temp_monitor._labels.append(lbl)
                    temp_monitor._groups.append(grp)
                # Replace the most-recent batch slice with fresh predictions
                n_batch = len(batch.X)
                temp_monitor._preds[-n_batch:] = new_preds_batch.tolist()
                new_snapshot = temp_monitor.snapshot(
                    batch.timestamp, float((new_preds_batch == batch.y).mean())
                )

                # Accept if utility is preserved AND SPD did not worsen on the
                # window-level estimate (low variance, apples-to-apples with snapshot).
                fairness_acceptable = (
                    new_snapshot.spd <= snapshot.spd + 0.02
                )

                if recovered and fairness_acceptable:
                    action = UnlearningAction(
                        timestamp=batch.timestamp,
                        candidates=candidates,
                        fairness_before=snapshot,
                        fairness_after=new_snapshot,
                        utility_before=old_acc,
                        utility_after=float((new_preds_batch == batch.y).mean()),
                        method=method,
                        cost_seconds=cost,
                        accepted=True,
                    )
                    self.audit.log(action)
                    self.detector.reset()  # cooldown
                    # Fine-tune on replay buffer for 2 steps to partially recover
                    # accuracy after the Newton step. Fewer steps than before
                    # (was 5) to avoid washing out the fairness gain: each
                    # fine-tuning step on biased replay data partially re-introduces
                    # the very bias we just removed.
                    self.preserver.fine_tune(self.replay_buffer, steps=2)
                    # Apply standard learning on this batch so the model
                    # does not skip training on accepted-unlearning batches.
                    self.model.update(batch.X, batch.y)
                else:
                    self.model.set_params(old_params)  # ROLLBACK
                    notes = (
                        "rollback: utility drop exceeded tolerance"
                        if not recovered
                        else "rollback: fairness worsened after unlearning"
                    )
                    action = UnlearningAction(
                        timestamp=batch.timestamp,
                        candidates=candidates,
                        fairness_before=snapshot,
                        fairness_after=new_snapshot,
                        utility_before=old_acc,
                        utility_after=float((self.model.predict_labels(batch.X) == batch.y).mean()),
                        method=method,
                        cost_seconds=cost,
                        accepted=False,
                        notes=notes,
                    )
                    self.audit.log(action)
                    # Unlearning was rejected; still apply standard online
                    # learning so the model does not stagnate when violations
                    # persist across many consecutive batches.
                    self.model.update(batch.X, batch.y)


                # Adaptive threshold
                self.cfg.fairness_threshold = self.audit.adaptive_threshold(
                    self.cfg.fairness_threshold
                )
                # Keep detector in sync with (possibly updated) threshold
                self.detector._threshold = self.cfg.fairness_threshold

            else:
                # Standard online learning
                self.model.update(batch.X, batch.y)

            # Always update replay buffer
            self.replay_buffer.add(batch)

        return history, self.audit.get_history()
