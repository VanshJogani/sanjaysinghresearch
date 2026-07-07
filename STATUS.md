# Project Status

**Last updated:** 2026-06-22

## Main Goal
Build a complete, defensible adaptive machine unlearning framework that:
1. Maintains fairness in a deployed model as new data streams in
2. Detects when bias re-emerges due to distribution shift or biased injections
3. Selectively removes problematic data influence without full retraining
4. Preserves model utility while doing so
5. Logs everything for accountability/reproducibility

**What it takes to achieve this:**
- Close the 8 implementation gaps between the plan PDFs and the code
- Add MLP experiments (non-convex model validation)
- Produce comparative evaluation results (AFU vs 3 baselines)
- Write up limitations clearly (Scenario A only, convexity assumption, etc.)

---

## Goals vs Completed

| # | Goal | Status | Notes |
|---|------|--------|-------|
| 1 | Data stream simulation (synthetic + real) | ✅ Done | Synthetic with drift/bias injection, COMPAS, Home Credit |
| 2 | Base model deployment | ⚠️ Partial | Model exists but no pre-training on initial batch (starts from zeros) |
| 3 | Online fairness monitoring (SPD, EOD) | ✅ Done | Sliding window, correct formulas |
| 4 | Bias detection trigger | ⚠️ Weak | Simple consecutive-count threshold. Plan says CUSUM/KS-test |
| 5 | Bias source identification (influence functions) | ✅ Done | LiSSA working, point-level attribution |
| 6 | Selective unlearning engine | ✅ Done | 3 mechanisms, auto-selection |
| 7 | Utility preservation | ⚠️ Partial | EWC + replay buffer. No knowledge distillation |
| 8 | Audit trail & adaptive control | ✅ Done | Logging, adaptive threshold feedback loop |
| 9 | Evaluation protocol & baselines | ✅ Done | 3 baselines, comparative evaluator |
| 10 | Drift detection (two-level trigger) | ❌ Missing | CRITICAL — without this, infinite loop on inherently biased data |
| 11 | Multi-level attribution (batch→point→feature) | ❌ Missing | Only point-level exists |
| 12 | Non-convex model (MLP) validation | ❌ Missing | Plan says to do both convex and non-convex |
| 13 | Multi-objective acceptance rule | ❌ Missing | Currently hardcoded thresholds |
| 14 | SISA fallback for large U | ❌ Missing | Plan mentions, not implemented |
| 15 | Knowledge distillation (L_KD) | ❌ Missing | Plan 2 proposes, not implemented |

---

## Vulnerabilities / Design Issues Found

1. **Infinite unlearning loop** — No drift detector means inherently biased data (COMPAS) triggers perpetual unlearning. System fights the base distribution forever.
2. **No pre-training** — Model starts at zeros, SPD is meaningless for first few batches. `consecutive_violations=7` is a hack workaround.
3. **Fairness gradient is a heuristic** — `grad_group0 - grad_group1` is NOT the derivative of SPD/EOD w.r.t. θ. The threshold/indicator function makes true differentiation non-trivial.
4. **Precondition not enforced** — System assumes initial fair deployment but has no mechanism to ensure or check this.
5. **One-directional alarm** — Only fires when fairness degrades, never when it improves. This is by design but means deploying biased → getting fair data won't help.
6. **Fixed LiSSA parameters** — 10 iterations, damping=0.01 regardless of problem scale. No convergence checking.

---

## Features to Add (Priority Order)

### High Priority (paper-blocking)
1. **Drift detector** — Two-level trigger: fairness violation AND evidence of distribution shift. Prevents infinite loop. Methods: ADWIN, PSI, KL divergence, or KS-test.
2. **Initial model pre-training** — Train on first batch before streaming loop starts. Ensures stable θ₀.
3. **Direction-of-change detection** — Detect "is SPD *increasing*?" not just "is SPD > threshold?"

### Medium Priority (strengthens paper)
4. **MLP model** — Demonstrate framework on non-convex model. Shows practical applicability beyond logistic regression.
5. **Batch-level screening** — Filter batches before expensive point-level influence computation.
6. **Feature-level attribution** — Detect when a feature becomes a proxy for protected attribute.
7. **Multi-objective acceptance rule** — J = β₁ΔFairness − β₂ΔError − β₃Cost instead of hardcoded thresholds.

### Lower Priority (nice-to-have)
8. **Knowledge distillation** — L_KD = E[D_KL(f_θ ∥ f_θ')] for better utility preservation.
9. **SISA fallback** — Sharded training for when |U| is very large.
10. **Adaptive windows** — Expand/contract monitoring window based on detected drift.
11. **CUSUM/KS-test** — Replace simple consecutive-count detector with statistical test.

---

## What "Done" Looks Like

- [ ] All 8 plan-vs-code gaps closed (or explicitly scoped out with justification)
- [ ] Drift detector prevents infinite loop on COMPAS without injection
- [ ] Model pre-trained before streaming loop
- [ ] Comparative results showing AFU vs baselines on both synthetic and real data
- [ ] MLP experiments showing framework generalizes beyond convex models
- [ ] Clear "Limitations" section articulating scope (Scenario A, convexity, etc.)
- [ ] Audit trail demonstrates accountability use case
