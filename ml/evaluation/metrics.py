from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]


def calculate_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> EvaluationResult:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
        "true_negatives": float(tn),
        "predicted_fraud_count": float(predictions.sum()),
    }
    return EvaluationResult(
        metrics=metrics,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )
