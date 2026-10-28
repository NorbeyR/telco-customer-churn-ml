"""Métricas de evaluación y costo de retención."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def total_cost(fn: int, fp: int, cost_fn: float, cost_fp: float) -> float:
    return float(fn * cost_fn + fp * cost_fp)


def compute_metrics(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    """Métricas sobre la clase Churn (1) con un umbral de decisión dado."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = total_cost(int(fn), int(fp), cost_fn, cost_fp)
    return {
        "umbral": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_churn": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_churn": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_churn": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "costo_total": cost,
        "costo_por_cliente": round(cost / len(y_true), 4),
    }
