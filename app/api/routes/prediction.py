from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.exceptions import ArtifactNotFoundError, DatasetValidationError
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["prediction"])
service = PredictionService()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return service.predict(request.features)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
