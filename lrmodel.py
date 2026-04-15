import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from home_credit_bias_demo import load_home_credit
from adaptive_fairness_unlearning.data.stream import RealWorldStreamLoader
from adaptive_fairness_unlearning.models.base_model import OnlineLogisticRegression
from adaptive_fairness_unlearning.monitors.fairness_monitor import FairnessMonitor

X, y, protected, time_order = load_home_credit("application_train.csv")
loader = RealWorldStreamLoader(X, y, protected, time_order)

model   = OnlineLogisticRegression(n_features=X.shape[1], lr=0.005)
monitor = FairnessMonitor(window_size=2500)

pos_rate     = y.mean()
class_weight = np.sqrt((1.0 - pos_rate) / pos_rate)
print(f"Positive rate: {pos_rate:.3f}  ->  class_weight = {class_weight:.2f}x\n")

L2 = 1e-4

_probs_buf, _y_buf = [], []

def _metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    _probs_buf.extend(y_prob.tolist())
    _y_buf.extend(y_true.tolist())
    if len(_probs_buf) > 5000:
        del _probs_buf[:-5000]
        del _y_buf[:-5000]

    try:
        auc = roc_auc_score(_y_buf, _probs_buf) if len(set(_y_buf)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    return accuracy, precision, recall, auc

# NOTE: Accuracy ~91.9% is the trivial all-zeros baseline (91.9% of rows are non-default).
# Precision/Recall will be 0 or NaN until MeanProb exceeds 0.5 for some samples.
# AUC is the honest discriminative metric — it does not depend on the threshold.
print(f"{'Batch':>6}  {'AUC':>6}  {'Accuracy':>9}  {'Precision':>10}  {'Recall':>7}  {'MeanProb':>9}  {'True1%':>7}  {'SPD':>7}")
print("-" * 78)

for batch in loader.stream(batch_size=500):
    y_prob = model.predict(batch.X)
    accuracy, precision, recall, auc = _metrics(batch.y, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    monitor.update(y_pred, batch.y, batch.protected)

    prec_str = f"{precision:10.3f}" if not np.isnan(precision) else f"{'N/A':>10}"
    print(f"{batch.timestamp:>6}  {auc:>6.3f}  {accuracy:>9.3f}  {prec_str}  "
          f"{recall:>7.3f}  {y_prob.mean():>9.4f}  {batch.y.mean():>7.3f}  {monitor.spd():>7.4f}")

    model.update(batch.X, batch.y, class_weight=class_weight)
    model.w *= (1.0 - model.lr * L2)
