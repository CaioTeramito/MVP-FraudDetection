from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

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
def thresholds(model: str | None = Query(default=None)) -> list[ThresholdTableRow]:
    try:
        artifact = get_predictor().artifact
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        rows = ThresholdService(artifact).for_model(model)["threshold_table"]
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [ThresholdTableRow(**row) for row in rows]


@router.post("/threshold/evaluate", response_model=ThresholdEvaluationResponse)
def threshold_evaluate(
    request: ThresholdEvaluationRequest,
    model: str | None = Query(default=None),
) -> ThresholdEvaluationResponse:
    try:
        artifact = get_predictor().artifact
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # This endpoint is exploratory only. The artifact's test arrays are used
    # to report evaluation metrics and never update the persisted threshold.
    # Threshold selection for training happens exclusively on validation data.
    service = ThresholdService(artifact)
    selected = service.for_model(model)
    threshold_artifacts = service.reevaluate(
        probabilities=selected["validation_probabilities"],
        y_true=selected["validation_truth"],
        model_name=model,
        step=request.step,
        minimum_precision=request.minimum_precision,
        minimum_recall=request.minimum_recall,
        cost_fn=request.cost_fn,
        cost_fp=request.cost_fp,
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
        evaluation_scope="validation_candidate" if model else "test_only_exploratory",
        rows=[
            ThresholdTableRow(**row)
            for row in threshold_artifacts.table.to_dict(orient="records")
        ],
    )
