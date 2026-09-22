from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ThresholdSummary(BaseModel):
    threshold: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    f2: float | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    dataset_name: str
    target_column: str
    feature_columns: list[str]
    threshold: float
    threshold_low: float
    threshold_high: float
    metrics: dict[str, float]
    training_config: dict[str, Any] = Field(default_factory=dict)
    best_thresholds: dict[str, ThresholdSummary] = Field(default_factory=dict)
