from __future__ import annotations

from pydantic import BaseModel


class MetricsResponse(BaseModel):
    model_name: str
    threshold: float
    metrics: dict[str, float]
