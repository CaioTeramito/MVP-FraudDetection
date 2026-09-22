from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ml.evaluation.metrics import calculate_classification_metrics


@dataclass(slots=True)
class ThresholdArtifacts:
    table: pd.DataFrame
    best_thresholds: dict[str, dict[str, float]]
    precision_recall_curve: dict[str, list[float]]
    roc_curve: dict[str, list[float]]


def generate_thresholds(step: float) -> np.ndarray:
    return np.round(np.arange(step, 1.0, step), 2)


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    step: float,
    minimum_precision: float,
    minimum_recall: float,
) -> ThresholdArtifacts:
    rows: list[dict[str, float]] = []
    for threshold in generate_thresholds(step):
        evaluation = calculate_classification_metrics(y_true, probabilities, float(threshold))
        row = {"threshold": float(threshold), **evaluation.metrics}
        rows.append(row)

    table = pd.DataFrame(rows)

    best_f1 = table.loc[table["f1"].idxmax()]
    best_f2 = table.loc[table["f2"].idxmax()]

    recall_candidates = table[table["precision"] >= minimum_precision]
    best_recall_constrained = (
        recall_candidates.sort_values(["recall", "threshold"], ascending=[False, True]).head(1)
        if not recall_candidates.empty
        else pd.DataFrame()
    )

    precision_candidates = table[table["recall"] >= minimum_recall]
    best_precision_constrained = (
        precision_candidates.sort_values(
            ["precision", "threshold"], ascending=[False, False]
        ).head(1)
        if not precision_candidates.empty
        else pd.DataFrame()
    )

    precision_values, recall_values, pr_thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )
    fpr_values, tpr_values, roc_thresholds = roc_curve(y_true, probabilities)

    best_thresholds = {
        "max_f1": {
            "threshold": float(best_f1["threshold"]),
            "precision": float(best_f1["precision"]),
            "recall": float(best_f1["recall"]),
            "f1": float(best_f1["f1"]),
            "f2": float(best_f1["f2"]),
        },
        "max_f2": {
            "threshold": float(best_f2["threshold"]),
            "precision": float(best_f2["precision"]),
            "recall": float(best_f2["recall"]),
            "f1": float(best_f2["f1"]),
            "f2": float(best_f2["f2"]),
        },
    }

    if not best_recall_constrained.empty:
        row = best_recall_constrained.iloc[0]
        best_thresholds["max_recall_with_min_precision"] = {
            "threshold": float(row["threshold"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "f2": float(row["f2"]),
        }

    if not best_precision_constrained.empty:
        row = best_precision_constrained.iloc[0]
        best_thresholds["max_precision_with_min_recall"] = {
            "threshold": float(row["threshold"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "f2": float(row["f2"]),
        }

    return ThresholdArtifacts(
        table=table,
        best_thresholds=best_thresholds,
        precision_recall_curve={
            "precision": precision_values.tolist(),
            "recall": recall_values.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        roc_curve={
            "fpr": fpr_values.tolist(),
            "tpr": tpr_values.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
    )
