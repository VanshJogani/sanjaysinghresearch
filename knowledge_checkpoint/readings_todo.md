# Readings & Study Topics

## Priority Order

1. **Koh & Liang 2017** — theoretical backbone of the entire system
2. **Bourtoule et al. 2021 (SISA)** — referenced in plan but not engaged with
3. **Any concept drift survey (Lu et al. 2018)** — biggest implementation gap

---

## 1. Differentiating fairness metrics w.r.t. model parameters

- **Gap:** Don't know why SPD = |P(ŷ=1|A=0) - P(ŷ=1|A=1)| is hard to differentiate w.r.t. θ
- **Key issue:** The indicator function (thresholding) inside P(ŷ=1|...) is non-differentiable. Common workarounds: use sigmoid probabilities instead of hard predictions, or use surrogate fairness losses.
- **Read:** "Fairness Constraints: Mechanisms for Fair Classification" (Zafar et al., 2017) — shows how to create differentiable fairness surrogates
- **Also read:** Section on "relaxed fairness constraints" in any fairness-in-ML survey

---

## 2. LiSSA convergence (Neumann series interpretation)

- **Gap:** Have rough intuition (each iteration adds a smaller term, (I-H)^10 becomes negligible) but can't derive it formally
- **Key insight:** If spectral radius of (I-H) < 1, then H⁻¹ = Σ_{k=0}^∞ (I-H)^k (Neumann series). LiSSA is unrolling this sum iteratively.
- **Read:** "Understanding Black-box Predictions via Influence Functions" (Koh & Liang, 2017) — Section 3 + Appendix on LiSSA
- **Also:** Linear algebra review on Neumann series for matrix inversion

---

## 3. Convexity and positive-definite Hessians

- **Gap:** Understand intuitively that convexity helps but can't articulate eigenvalue conditions precisely
- **Key insight:** Convex loss → H is PSD → eigenvalues ≥ 0 → LiSSA converges, Taylor approximation valid globally (not just locally). Non-convex (neural nets) → H can have negative eigenvalues → divergence risk, approximation only valid near current θ.
- **Read:** Any ML optimization textbook on convexity (Boyd & Vandenberghe Ch. 3-4), or Bishop Pattern Recognition Ch. 4 (logistic regression as convex problem)

---

## 4. Drift detection methods (ADWIN, PSI, KS-test, CUSUM)

- **Gap:** Not explored at all — know it's needed (two-level trigger) but haven't studied the methods
- **Read:** "Learning under Concept Drift: A Review" (Lu et al., 2018) — covers ADWIN, DDM, EDDM
- **Also:** River library documentation on drift detectors (practical implementation reference)

---

## 5. Knowledge distillation for utility preservation

- **Gap:** Plan 2 proposes L_KD = E[D_KL(f_θ ∥ f_θ')], not implemented, not understood
- **Read:** "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
- **Context:** In unlearning, you use KD to keep the post-unlearning model's outputs close to the pre-unlearning model on retained data

---

## 6. SISA (Sharded, Isolated, Sliced, Aggregated) unlearning

- **Gap:** Plan 1 mentions as fallback for large U, not implemented, not discussed
- **Read:** "Machine Unlearning" (Bourtoule et al., 2021) — introduces SISA
- **Key idea:** Split training data into shards, train sub-models on each shard independently, aggregate predictions. Unlearning only requires retraining the affected shard.
