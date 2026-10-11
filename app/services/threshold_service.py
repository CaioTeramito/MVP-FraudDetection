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

    def reevaluate(
        self,
        probabilities: list[float],
        y_true: list[int],
        *,
        step: float | None = None,
        minimum_precision: float | None = None,
        minimum_recall: float | None = None,
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
        return evaluate_thresholds(
            y_true=pd.Series(y_true).to_numpy(),
            probabilities=pd.Series(probabilities).to_numpy(),
            step=effective_step,
            minimum_precision=effective_minimum_precision,
            minimum_recall=effective_minimum_recall,
        )
