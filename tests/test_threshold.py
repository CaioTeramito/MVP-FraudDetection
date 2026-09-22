from __future__ import annotations

import numpy as np

from ml.threshold.analysis import evaluate_thresholds


def test_threshold_analysis_returns_expected_columns():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.40, 0.55, 0.80, 0.95])
    result = evaluate_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        step=0.05,
        minimum_precision=0.5,
        minimum_recall=0.5,
    )
    assert not result.table.empty
    assert {"precision", "recall", "f1", "f2", "false_positive_rate"}.issubset(
        result.table.columns
    )
    assert "max_f2" in result.best_thresholds
