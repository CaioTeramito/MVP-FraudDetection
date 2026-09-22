from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Flexible transaction schema driven by artifact feature metadata."""

    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any] = Field(
        ...,
        description="Feature map for a single transaction using dataset column names.",
    )


class FeatureContribution(BaseModel):
    feature: str
    value: Any
    contribution: float


class PredictResponse(BaseModel):
    fraud_probability: float
    threshold: float
    prediction: int
    risk: str
    decision: str
    model_name: str
    model_version: str
    top_contributors: list[FeatureContribution] = Field(default_factory=list)
