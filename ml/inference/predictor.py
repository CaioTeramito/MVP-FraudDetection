from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import ArtifactNotFoundError
from app.services.decision import DecisionService
from ml.data.dataset import validate_feature_payload
from ml.inference.artifact import ModelArtifact, load_artifact

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


class FraudPredictor:
    """Loads a serialized artifact and serves inference requests."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact
        self.decision_service = DecisionService(
            threshold_low=artifact.threshold_low,
            threshold_high=artifact.threshold_high,
        )

    @classmethod
    def from_path(cls, path: str | None = None) -> "FraudPredictor":
        artifact_path = path or str(settings.model_artifact_path)
        if not pd.io.common.file_exists(artifact_path):
            raise ArtifactNotFoundError(
                f"Artefato do modelo nao encontrado em '{artifact_path}'. Execute o treinamento primeiro."
            )
        return cls(load_artifact(artifact_path))

    def predict_transaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        validate_feature_payload(payload, self.artifact.feature_columns)
        features = pd.DataFrame([payload], columns=self.artifact.feature_columns)
        transformed = self.artifact.preprocessor.transform(features)
        probability = float(self.artifact.estimator.predict_proba(transformed)[:, 1][0])
        prediction = int(probability >= self.artifact.threshold)
        decision = self.decision_service.decide(probability)

        return {
            "fraud_probability": probability,
            "threshold": self.artifact.threshold,
            "prediction": prediction,
            "risk": decision.risk,
            "decision": decision.decision,
            "model_name": self.artifact.model_name,
            "model_version": self.artifact.model_version,
            "top_contributors": self.explain(features, transformed),
        }

    def explain(self, original_features: pd.DataFrame, transformed: Any) -> list[dict[str, Any]]:
        if not self.artifact.shap_available or shap is None:
            return []

        estimator = self.artifact.estimator
        try:
            explainer = shap.Explainer(estimator)
            shap_values = explainer(transformed)
            contributions = np.asarray(shap_values.values[0]).ravel()
            paired = list(
                zip(self.artifact.transformed_feature_names, contributions, strict=False)
            )
            top = sorted(paired, key=lambda item: abs(item[1]), reverse=True)[:5]
            value_lookup = original_features.iloc[0].to_dict()
            explanation = []
            for feature_name, contribution in top:
                raw_feature = feature_name.split("__")[-1].split("_")[0]
                explanation.append(
                    {
                        "feature": feature_name,
                        "value": value_lookup.get(raw_feature),
                        "contribution": float(contribution),
                    }
                )
            return explanation
        except Exception:
            return []
