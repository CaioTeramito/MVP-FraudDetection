from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ThresholdStrategy(ABC):
    """Common interface for selecting one binary fraud threshold."""

    name: str

    @abstractmethod
    def select(self, table: pd.DataFrame) -> pd.Series | None:
        """Return the selected row, or None when constraints have no solution."""


class MaxF1Strategy(ThresholdStrategy):
    name = "max_f1"

    def select(self, table: pd.DataFrame) -> pd.Series:
        return table.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]


class MaxF2Strategy(ThresholdStrategy):
    name = "max_f2"

    def select(self, table: pd.DataFrame) -> pd.Series:
        return table.sort_values(["f2", "threshold"], ascending=[False, True]).iloc[0]


class MaxRecallWithMinPrecisionStrategy(ThresholdStrategy):
    name = "max_recall_with_min_precision"

    def __init__(self, minimum_precision: float) -> None:
        self.minimum_precision = _validate_rate(minimum_precision, "minimum_precision")

    def select(self, table: pd.DataFrame) -> pd.Series | None:
        candidates = table[table["precision"] >= self.minimum_precision]
        if candidates.empty:
            return None
        return candidates.sort_values(
            ["recall", "threshold"], ascending=[False, True]
        ).iloc[0]


class MaxPrecisionWithMinRecallStrategy(ThresholdStrategy):
    name = "max_precision_with_min_recall"

    def __init__(self, minimum_recall: float) -> None:
        self.minimum_recall = _validate_rate(minimum_recall, "minimum_recall")

    def select(self, table: pd.DataFrame) -> pd.Series | None:
        candidates = table[table["recall"] >= self.minimum_recall]
        if candidates.empty:
            return None
        return candidates.sort_values(
            ["precision", "threshold"], ascending=[False, False]
        ).iloc[0]


class CostThresholdStrategy(ThresholdStrategy):
    """Select a binary threshold by minimizing expected validation cost.

    In the default mode the formula is exactly ``c_fn * FN + c_fp * FP``.
    With ``amount_weighted=True``, the FN term becomes ``c_fn`` times the sum
    of transaction amounts for missed frauds; FP remains count-based.
    """

    name = "cost"

    def __init__(
        self,
        cost_fn: float,
        cost_fp: float,
        *,
        amount_weighted: bool = False,
    ) -> None:
        self.cost_fn = _validate_non_negative(cost_fn, "cost_fn")
        self.cost_fp = _validate_non_negative(cost_fp, "cost_fp")
        self.amount_weighted = amount_weighted

    def add_cost_columns(
        self,
        table: pd.DataFrame,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        amount: np.ndarray | None = None,
    ) -> pd.DataFrame:
        y_true, probabilities = _validate_observations(y_true, probabilities)
        if self.amount_weighted:
            if amount is None:
                raise ValueError("amount is required when amount_weighted=True")
            amount = np.asarray(amount, dtype=float)
            if amount.ndim != 1 or len(amount) != len(y_true):
                raise ValueError("amount must have the same number of samples as y_true")
            if not np.isfinite(amount).all() or (amount < 0).any():
                raise ValueError("amount must contain finite non-negative values")

        result = table.copy()
        expected_costs: list[float] = []
        fn_costs: list[float] = []
        fp_costs: list[float] = []
        for threshold in result["threshold"].to_numpy(dtype=float):
            predictions = probabilities >= threshold
            false_negative_mask = (y_true == 1) & ~predictions
            false_positive_mask = (y_true == 0) & predictions
            if self.amount_weighted:
                fn_cost = self.cost_fn * float(amount[false_negative_mask].sum())
            else:
                fn_cost = self.cost_fn * int(false_negative_mask.sum())
            fp_cost = self.cost_fp * int(false_positive_mask.sum())
            expected_costs.append(fn_cost + fp_cost)
            fn_costs.append(fn_cost)
            fp_costs.append(fp_cost)
        result["false_negative_cost"] = fn_costs
        result["false_positive_cost"] = fp_costs
        result["expected_cost"] = expected_costs
        return result

    def select(self, table: pd.DataFrame) -> pd.Series:
        if "expected_cost" not in table.columns:
            raise ValueError("Cost table must contain expected_cost")
        # Ties use the smallest threshold, making the choice deterministic.
        return table.sort_values(
            ["expected_cost", "threshold"], ascending=[True, True]
        ).iloc[0]


def build_threshold_strategy(
    name: str,
    *,
    minimum_precision: float,
    minimum_recall: float,
    cost_fn: float | None = None,
    cost_fp: float | None = None,
    amount_weighted: bool = False,
) -> ThresholdStrategy:
    if name == "max_f1":
        return MaxF1Strategy()
    if name == "max_f2":
        return MaxF2Strategy()
    if name == "max_recall_with_min_precision":
        return MaxRecallWithMinPrecisionStrategy(minimum_precision)
    if name == "max_precision_with_min_recall":
        return MaxPrecisionWithMinRecallStrategy(minimum_recall)
    if name == "cost":
        if cost_fn is None or cost_fp is None:
            raise ValueError("cost_fn and cost_fp are required for the cost strategy")
        return CostThresholdStrategy(
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            amount_weighted=amount_weighted,
        )
    raise ValueError(f"Unknown threshold strategy: {name}")


def strategy_parameters(strategy: ThresholdStrategy) -> dict[str, Any]:
    if isinstance(strategy, CostThresholdStrategy):
        return {
            "cost_fn": strategy.cost_fn,
            "cost_fp": strategy.cost_fp,
            "amount_weighted": strategy.amount_weighted,
        }
    if isinstance(strategy, MaxRecallWithMinPrecisionStrategy):
        return {"minimum_precision": strategy.minimum_precision}
    if isinstance(strategy, MaxPrecisionWithMinRecallStrategy):
        return {"minimum_recall": strategy.minimum_recall}
    return {}


def _validate_rate(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _validate_non_negative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_observations(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional")
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if len(y_true) == 0:
        raise ValueError("y_true and probabilities cannot be empty")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    return y_true.astype(int), probabilities
