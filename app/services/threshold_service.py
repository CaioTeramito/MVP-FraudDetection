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
        return evaluate_thresholds(
            y_true=pd.Series(y_true).to_numpy(),
            probabilities=pd.Series(probabilities).to_numpy(),
            step=step or settings.threshold_step,
            minimum_precision=minimum_precision or settings.minimum_precision,
            minimum_recall=minimum_recall or settings.minimum_recall,
        )
