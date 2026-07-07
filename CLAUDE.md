# Adaptive Fairness Unlearning (AFU) Framework

## What This Project Is
A Python framework for adaptive machine unlearning that monitors a deployed model's fairness metrics in real-time, detects bias re-emergence from distribution shifts or biased data injections, and selectively unlearns problematic data points without full retraining.

**Scope:** Solves bias *re-emergence* in a previously fair model (Scenario A). Does NOT solve inherent/static bias in historical data (Scenario B).

## How to Run

```bash
# Synthetic data demo (quickest way to see the full pipeline)
python run_demo.py

# COMPAS recidivism dataset demo
python compas_demo.py

# COMPAS with injected bias
python compas_bias_demo.py

# Home Credit with injected bias
python home_credit_bias_demo.py
```

Datasets required: `compas-scores-raw.csv`, `application_train.csv` (in project root).

## Architecture

```
adaptive_fairness_unlearning/
├── pipeline.py          # Main loop (Algorithm 1) — orchestrates everything
├── models/base_model.py # OnlineLogisticRegression with gradient/HVP support
├── monitors/fairness_monitor.py  # Sliding window SPD/EOD computation
├── detectors/bias_detector.py    # Consecutive-threshold trigger
├── attribution/influence.py      # LiSSA-based influence estimation
├── unlearning/engine.py          # 3 mechanisms: Newton, gradient reversal, reweight
├── utility/preservation.py       # EWC recovery + replay buffer
├── audit/logger.py               # Event logging + adaptive threshold
└── evaluation/benchmarks.py      # 3 baselines + comparative evaluator
```

**Flow:** stream input → predict → monitor fairness → detect violation → attribute via influence → selective unlearn → utility check → accept/rollback → log

## Key Technical Details
- Model: Online logistic regression (convex — makes influence functions valid)
- Influence approximation: LiSSA (10 iterations, damping=0.01)
- Unlearning method auto-selected by |U|: ≤20 Newton, 20-100 gradient reversal, >100 reweight
- Fairness metrics: SPD (Statistical Parity Difference), EOD (Equalized Odds Difference)
- Online learning: model.update() runs on EVERY batch (line 224 of pipeline.py)

## Conventions
- NumPy throughout (no PyTorch/JAX yet)
- Config via `FrameworkConfig` dataclass
- All modules take model reference at init
- Protected attribute is binary (0/1)

## Known Critical Gaps (see STATUS.md for full list)
- No pre-training on initial batch (model starts from zeros)
- No drift detector (triggers unlearning on inherently biased data → infinite loop)
- Fairness gradient is a heuristic, not true ∇θ SPD
