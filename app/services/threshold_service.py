from __future__ import annotations

import pandas as pd

from app.core.config import settings
from ml.inference.artifact import ModelArtifact
from ml.threshold.analysis import evaluate_thresholds


class ThresholdService:
    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.artifact.threshold_table)

    def for_model(self, model_name: str | None = None) -> dict:
        name = model_name or self.artifact.model_name
        if model_name is None or not self.artifact.model_results:
            return {
                "model_name": self.artifact.model_name,
                "threshold_table": self.artifact.threshold_table,
                "validation_truth": self.artifact.holdout_truth,
                "validation_probabilities": self.artifact.holdout_probabilities,
            }
        try:
            return self.artifact.model_results[name]
        except KeyError as exc:
            raise ValueError(f"Modelo '{name}' não encontrado no artefato") from exc

    def reevaluate(
        self,
        probabilities: list[float],
        y_true: list[int],
        *,
        step: float | None = None,
        minimum_precision: float | None = None,
        minimum_recall: float | None = None,
        model_name: str | None = None,
        cost_fn: float | None = None,
        cost_fp: float | None = None,
        amount: list[float] | None = None,
    ):
        effective_step = settings.threshold_step if step is None else step
        effective_minimum_precision = (
            settings.minimum_precision
            if minimum_precision is None
            else minimum_precision
        )
        effective_minimum_recall = (
            settings.minimum_recall if minimum_recall is None else minimum_recall
        )
        selected = self.for_model(model_name)
        effective_probabilities = probabilities or selected.get("validation_probabilities", [])
        effective_y_true = y_true or selected.get("validation_truth", [])
        strategy = None
        if cost_fn is not None or cost_fp is not None:
            from ml.threshold.strategies import build_threshold_strategy
            if cost_fn is None or cost_fp is None:
                raise ValueError("cost_fn e cost_fp devem ser informados juntos")
            strategy = build_threshold_strategy("cost", cost_fn=cost_fn, cost_fp=cost_fp)
        return evaluate_thresholds(
            y_true=pd.Series(effective_y_true).to_numpy(),
            probabilities=pd.Series(effective_probabilities).to_numpy(),
            step=effective_step,
            minimum_precision=effective_minimum_precision,
            minimum_recall=effective_minimum_recall,
            strategy=strategy,
            amount=None if amount is None else pd.Series(amount).to_numpy(),
        )
