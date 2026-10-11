from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.threshold.analysis import evaluate_thresholds
from ml.threshold.strategies import CostThresholdStrategy


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


def test_cost_strategy_uses_count_formula_and_selects_lowest_cost():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.20, 0.80, 0.40, 0.90])
    strategy = CostThresholdStrategy(cost_fn=10, cost_fp=2)

    result = evaluate_thresholds(
        y_true=y_true,
        probabilities=probabilities,
        step=0.1,
        minimum_precision=0.5,
        minimum_recall=0.5,
        strategy=strategy,
    )

    assert result.best_thresholds["cost"]["threshold"] == pytest.approx(0.3)
    assert result.best_thresholds["cost"]["expected_cost"] == pytest.approx(2.0)


def test_cost_strategy_supports_amount_weighted_false_negatives():
    y_true = np.array([1, 1, 0])
    probabilities = np.array([0.10, 0.80, 0.90])
    amount = np.array([100.0, 1000.0, 10.0])
    strategy = CostThresholdStrategy(cost_fn=1, cost_fp=5, amount_weighted=True)
    table = pd.DataFrame({"threshold": [0.5, 0.95]})

    result = strategy.add_cost_columns(table, y_true, probabilities, amount)

    assert result.loc[0, "false_negative_cost"] == pytest.approx(100.0)
    assert result.loc[1, "false_negative_cost"] == pytest.approx(1100.0)
    assert result.loc[0, "false_positive_cost"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cost_fn": -1, "cost_fp": 1},
        {"cost_fn": 1, "cost_fp": -1},
    ],
)
def test_cost_strategy_rejects_negative_costs(kwargs):
    with pytest.raises(ValueError):
        CostThresholdStrategy(**kwargs)


def test_cost_strategy_rejects_invalid_amount_shape():
    strategy = CostThresholdStrategy(cost_fn=1, cost_fp=1, amount_weighted=True)
    table = pd.DataFrame({"threshold": [0.5]})
    with pytest.raises(ValueError):
        strategy.add_cost_columns(table, np.array([1, 0]), np.array([0.8, 0.2]), np.array([10]))
