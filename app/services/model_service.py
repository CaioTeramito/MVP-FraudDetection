from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from ml.inference.predictor import FraudPredictor


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    return FraudPredictor.from_path(str(settings.model_artifact_path))


def reload_predictor() -> FraudPredictor:
    get_predictor.cache_clear()
    return get_predictor()
