from __future__ import annotations

from app.schemas.prediction import PredictResponse
from app.services.model_service import get_predictor


class PredictionService:
    def predict(self, features: dict[str, object]) -> PredictResponse:
        predictor = get_predictor()
        payload = predictor.predict_transaction(features)
        return PredictResponse(**payload)
