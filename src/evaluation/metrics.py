from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def calculate_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: Iterable[float] | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics used across experiments."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics
