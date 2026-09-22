from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DecisionResult:
    risk: str
    decision: str


class DecisionService:
    """Applies demonstration business rules on top of the model score."""

    def __init__(self, threshold_low: float, threshold_high: float) -> None:
        if threshold_low >= threshold_high:
            raise ValueError("threshold_low must be lower than threshold_high")
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def decide(self, probability: float) -> DecisionResult:
        if probability < self.threshold_low:
            return DecisionResult(risk="LOW", decision="APPROVE")
        if probability < self.threshold_high:
            return DecisionResult(risk="MEDIUM", decision="REVIEW")
        return DecisionResult(risk="HIGH", decision="BLOCK")
