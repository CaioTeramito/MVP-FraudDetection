from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.exceptions import ArtifactNotFoundError
from app.schemas.threshold import (
    ThresholdEvaluationRequest,
    ThresholdEvaluationResponse,
    ThresholdTableRow,
)
from app.services.model_service import get_predictor
from app.services.threshold_service import ThresholdService

router = APIRouter(tags=["threshold"])


@router.get("/thresholds", response_model=list[ThresholdTableRow])
def thresholds() -> list[ThresholdTableRow]:
    try:
        artifact = get_predictor().artifact
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [ThresholdTableRow(**row) for row in artifact.threshold_table]


@router.post("/threshold/evaluate", response_model=ThresholdEvaluationResponse)
def threshold_evaluate(
    request: ThresholdEvaluationRequest,
) -> ThresholdEvaluationResponse:
    try:
        artifact = get_predictor().artifact
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    service = ThresholdService(artifact)
    threshold_artifacts = service.reevaluate(
        probabilities=artifact.holdout_probabilities,
        y_true=artifact.holdout_truth,
        step=request.step,
        minimum_precision=request.minimum_precision,
        minimum_recall=request.minimum_recall,
    )
    if request.criterion not in threshold_artifacts.best_thresholds:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Criterio '{request.criterion}' indisponivel para os parametros informados. "
                f"Disponiveis: {sorted(threshold_artifacts.best_thresholds)}"
            ),
        )

    selected_threshold = threshold_artifacts.best_thresholds[request.criterion]["threshold"]
    return ThresholdEvaluationResponse(
        selected_threshold=selected_threshold,
        criterion=request.criterion,
        rows=[
            ThresholdTableRow(**row)
            for row in threshold_artifacts.table.to_dict(orient="records")
        ],
    )
