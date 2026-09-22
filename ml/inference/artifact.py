from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(slots=True)
class ModelArtifact:
    model_name: str
    model_version: str
    trained_at: str
    dataset_name: str
    dataset_path: str
    target_column: str
    feature_columns: list[str]
    transformed_feature_names: list[str]
    threshold: float
    threshold_low: float
    threshold_high: float
    metrics: dict[str, float]
    threshold_table: list[dict[str, float]]
    best_thresholds: dict[str, dict[str, float]]
    holdout_truth: list[int]
    holdout_probabilities: list[float]
    training_config: dict[str, Any]
    preprocessor: Any
    estimator: Any
    shap_available: bool
    notes: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "transformed_feature_names": self.transformed_feature_names,
            "threshold": self.threshold,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "metrics": self.metrics,
            "threshold_table": self.threshold_table,
            "best_thresholds": self.best_thresholds,
            "holdout_truth": self.holdout_truth,
            "holdout_probabilities": self.holdout_probabilities,
            "training_config": self.training_config,
            "preprocessor": self.preprocessor,
            "estimator": self.estimator,
            "shap_available": self.shap_available,
            "notes": self.notes,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelArtifact":
        return cls(**payload)

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        model_version: str,
        dataset_name: str,
        dataset_path: str,
        target_column: str,
        feature_columns: list[str],
        transformed_feature_names: list[str],
        threshold: float,
        threshold_low: float,
        threshold_high: float,
        metrics: dict[str, float],
        threshold_table: pd.DataFrame,
        best_thresholds: dict[str, dict[str, float]],
        holdout_truth: list[int],
        holdout_probabilities: list[float],
        training_config: dict[str, Any],
        preprocessor: Any,
        estimator: Any,
        shap_available: bool,
        notes: list[str] | None = None,
    ) -> "ModelArtifact":
        return cls(
            model_name=model_name,
            model_version=model_version,
            trained_at=datetime.now(timezone.utc).isoformat(),
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            target_column=target_column,
            feature_columns=feature_columns,
            transformed_feature_names=transformed_feature_names,
            threshold=threshold,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            metrics=metrics,
            threshold_table=threshold_table.to_dict(orient="records"),
            best_thresholds=best_thresholds,
            holdout_truth=holdout_truth,
            holdout_probabilities=holdout_probabilities,
            training_config=training_config,
            preprocessor=preprocessor,
            estimator=estimator,
            shap_available=shap_available,
            notes=notes or [],
        )


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact.to_payload(), destination)


def load_artifact(path: str | Path) -> ModelArtifact:
    payload = joblib.load(path)
    return ModelArtifact.from_payload(payload)
