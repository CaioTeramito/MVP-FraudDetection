from __future__ import annotations

from ml.inference.predictor import FraudPredictor


def test_predictor_returns_business_decision(sample_artifact):
    predictor = FraudPredictor.from_path(str(sample_artifact))
    result = predictor.predict_transaction({"feature_1": 0.95, "feature_2": 1})
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert result["risk"] in {"LOW", "MEDIUM", "HIGH"}
