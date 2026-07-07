# Knowledge Checkpoint — 2026-06-22

## First Deep Quiz Session

---

## Understanding Level

| Concept | Level | Notes |
|---------|-------|-------|
| Pipeline architecture & flow | ✅ Solid | Know what each module does and how they connect |
| Why H⁻¹ is approximated (O(p³) vs O(p²)) | ✅ Solid | Direct inversion computationally infeasible |
| LiSSA mechanics (Neumann series) | ⚠️ Rough | Got it with hints about geometric series; needs formal study |
| Newton-step unlearning logic | ✅ Got it | "Training adds effect, unlearning subtracts it, double negative = +" |
| When approximation breaks (large U) | ✅ Got it | Large U → big parameter shift → far from where H was computed |
| Infinite loop failure (no drift detector) | ✅ Reasoned through | Inherently biased data triggers perpetual unlearning cycle |
| Scope: dynamic bias vs static bias | ✅ Got it | System solves Scenario A (bias re-emergence), not Scenario B (inherent bias) |
| Online learning assumption | ✅ Now aware | Didn't know model.update() runs every batch; now understand |
| Model not pre-trained (plan vs code gap) | ✅ Now aware | Zeros init → SPD spike at start; consecutive_violations=7 is a hack |
| Why convexity matters | ⚠️ Got with guidance | PSD Hessian → LiSSA converges, Taylor approx valid globally |
| Fairness gradient differentiability | ❌ Needs reading | Don't know why SPD is non-differentiable (indicator/threshold) |
| Knowledge distillation (L_KD) | ❌ Not explored | Plan 2 proposes it, not implemented, not discussed |
| SISA / sharded unlearning | ❌ Not explored | Plan 1 mentions as fallback for large U, not implemented |
| Feature-level attribution | ❌ Not explored | Plan 2 has 3-level hierarchy, only point-level in code |
| Drift detection (PSI, ADWIN, KL) | ❌ Not explored | Critical gap — needed for two-level trigger |

---

## Key Gaps Between Plan and Code

1. **No pre-training on initial batch** — Plan Step 2 says to do this, code starts from zeros
2. **No drift detector** — Plan says two-level trigger (fairness violation AND drift), code only checks fairness
3. **Bias detector is simple consecutive-count** — Not CUSUM or KS-test as planned
4. **Only point-level attribution** — No batch screening or feature attribution (plan has 3 levels)
5. **No knowledge distillation** — For utility preservation
6. **No SISA fallback** — For large unlearning sets
7. **No multi-objective acceptance rule** — J = β₁ΔFairness − β₂ΔError − β₃Cost
8. **Fairness gradient is a heuristic** — Group difference, not true ∇θ SPD

---

## Design Decisions Made

- **Scope:** Framework targets bias re-emergence in previously fair models (Scenario A)
- **Precondition:** Model must be deployed fair; will add initial debiasing step
- **Detection improvement:** Will add direction-of-change detection
- **Non-convex validation:** Should add MLP experiments alongside logistic regression

---

## Questions I Asked That Were Good

- "What's the point if data is always inherently biased?" → Led to scope clarification
- "Does online learning even happen in practice?" → Critical deployment assumption
- "What model am I training? I don't remember pre-training it" → Found plan/code mismatch
- "If I deploy biased and fair data arrives, why doesn't fairness become normal?" → Understood asymmetric alarm

---

## Next Steps

1. Fix code gaps (pre-training, drift detection)
2. Complete readings (fairness gradients, Neumann series, Koh & Liang 2017)
3. Quiz on unexplored modules (KD, SISA, feature attribution)
4. Discuss experimental design — what results to produce
5. MLP experiments
