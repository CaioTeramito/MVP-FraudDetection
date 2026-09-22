from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.exceptions import ArtifactNotFoundError
from app.schemas.model_info import ModelInfoResponse, ThresholdSummary
from app.schemas.metrics import MetricsResponse
from app.services.model_service import get_predictor

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        artifact = get_predictor().artifact
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    summaries = {
        name: ThresholdSummary(**values) for name, values in artifact.best_thresholds.items()
    }
    return ModelInfoResponse(
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        trained_at=artifact.trained_at,
        dataset_name=artifact.dataset_name,
        target_column=artifact.target_column,
        feature_columns=artifact.feature_columns,
        threshold=artifact.threshold,
        threshold_low=artifact.threshold_low,
        threshold_high=artifact.threshold_high,
        metrics=artifact.metrics,
        training_config=artifact.training_config,
        best_thresholds=summaries,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    try:
        artifact = get_predictor().artifact
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MetricsResponse(
        model_name=artifact.model_name,
        threshold=artifact.threshold,
        metrics=artifact.metrics,
    )
