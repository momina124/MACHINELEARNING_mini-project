"""
Model Evaluation and Metric Analysis Module.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix,
and optimal business-cost decision threshold.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for a binary classifier.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_pred_proba)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # ROC curve points
    fpr, tpr, roc_thresh = roc_curve(y_true, y_pred_proba)
    # PR curve points
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_pred_proba)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'threshold': float(threshold),
        'confusion_matrix': cm,
        'classification_report': clf_report,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresh.tolist()
        },
        'pr_curve': {
            'precision': pr_prec.tolist(),
            'recall': pr_rec.tolist(),
            'thresholds': pr_thresh.tolist()
        }
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_fn: float = 150.0,
    cost_fp: float = 25.0
) -> Tuple[float, pd.DataFrame]:
    """
    Sweeps thresholds from 0.05 to 0.95 to find the optimal decision threshold
    balancing retention campaign costs (FP) against lost customer revenue (FN).
    Also records F1, F2 (recall-weighted), precision, and recall.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []

    best_thresh = 0.5
    min_cost = float('inf')

    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        # F2 score: gives 2x more weight to recall than precision
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

        total_cost = (fn * cost_fn) + (fp * cost_fp)

        records.append({
            'threshold': float(round(t, 2)),
            'precision': float(round(prec, 4)),
            'recall': float(round(rec, 4)),
            'f1_score': float(round(f1, 4)),
            'f2_score': float(round(f2, 4)),
            'total_cost': float(total_cost),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        })

        if total_cost < min_cost:
            min_cost = total_cost
            best_thresh = float(round(t, 2))

    df_sweep = pd.DataFrame(records)
    return best_thresh, df_sweep


def compare_models(evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a clean comparison DataFrame across models.
    """
    rows = []
    for model_name, metrics in evaluations.items():
        rows.append({
            'Model': model_name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall (Churn)': round(metrics['recall'], 4),
            'F1-Score': round(metrics['f1_score'], 4),
            'ROC-AUC': round(metrics['roc_auc'], 4),
            'PR-AUC': round(metrics['pr_auc'], 4),
            'Decision Threshold': metrics.get('threshold', 0.5)
        })
    df_compare = pd.DataFrame(rows).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    return df_compare
