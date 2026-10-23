from __future__ import annotations

from pydantic import BaseModel, Field


class ThresholdEvaluationRequest(BaseModel):
    criterion: str = Field(
        default="max_f2",
        description=(
            "Selection criterion: max_f1, max_f2, max_recall_with_min_precision, "
            "max_precision_with_min_recall."
        ),
    )
    minimum_precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Business constraint for recall-oriented threshold selection.",
    )
    minimum_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Business constraint for precision-oriented threshold selection.",
    )
    step: float | None = Field(default=None, gt=0.0, le=0.5)
    model_name: str | None = Field(default=None, description="Candidate model stored in the artifact.")
    cost_fn: float | None = Field(default=None, ge=0.0)
    cost_fp: float | None = Field(default=None, ge=0.0)


class ThresholdTableRow(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    roc_auc: float
    pr_auc: float
    false_positive_rate: float
    false_negative_rate: float
    false_positives: int
    false_negatives: int
    predicted_fraud_count: int
    expected_cost: float | None = None
    false_negative_cost: float | None = None
    false_positive_cost: float | None = None


class ThresholdEvaluationResponse(BaseModel):
    selected_threshold: float
    criterion: str
    rows: list[ThresholdTableRow]
    evaluation_scope: str = "test_only_exploratory"
