from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ml.evaluation.metrics import calculate_classification_metrics
from ml.threshold.strategies import (
    CostThresholdStrategy,
    MaxF1Strategy,
    MaxF2Strategy,
    MaxPrecisionWithMinRecallStrategy,
    MaxRecallWithMinPrecisionStrategy,
    ThresholdStrategy,
)


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
    strategy: ThresholdStrategy | None = None,
    amount: np.ndarray | None = None,
) -> ThresholdArtifacts:
    rows: list[dict[str, float]] = []
    for threshold in generate_thresholds(step):
        evaluation = calculate_classification_metrics(y_true, probabilities, float(threshold))
        row = {"threshold": float(threshold), **evaluation.metrics}
        rows.append(row)

    table = pd.DataFrame(rows)

    strategies: list[ThresholdStrategy] = [
        MaxF1Strategy(),
        MaxF2Strategy(),
        MaxRecallWithMinPrecisionStrategy(minimum_precision),
        MaxPrecisionWithMinRecallStrategy(minimum_recall),
    ]
    if isinstance(strategy, CostThresholdStrategy):
        table = strategy.add_cost_columns(table, y_true, probabilities, amount)
        strategies.append(strategy)

    precision_values, recall_values, pr_thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )
    fpr_values, tpr_values, roc_thresholds = roc_curve(y_true, probabilities)

    best_thresholds: dict[str, dict[str, float]] = {}
    for threshold_strategy in strategies:
        row = threshold_strategy.select(table)
        if row is None:
            continue
        selected = {
            "threshold": float(row["threshold"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "f2": float(row["f2"]),
        }
        if isinstance(threshold_strategy, CostThresholdStrategy):
            selected["expected_cost"] = float(row["expected_cost"])
            selected.update(
                {
                    "false_negative_cost": float(row["false_negative_cost"]),
                    "false_positive_cost": float(row["false_positive_cost"]),
                }
            )
        best_thresholds[threshold_strategy.name] = selected

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
