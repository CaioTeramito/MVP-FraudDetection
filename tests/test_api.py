from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "logistic_regression"


def test_predict_endpoint():
    response = client.post(
        "/api/v1/predict",
        json={"features": {"feature_1": 0.95, "feature_2": 1}},
    )
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body


def test_predict_rejects_invalid_payload():
    response = client.post(
        "/api/v1/predict",
        json={"features": {"feature_1": 0.95}},
    )
    assert response.status_code == 422


def test_threshold_evaluate_endpoint():
    response = client.post(
        "/api/v1/threshold/evaluate",
        json={"criterion": "max_f2", "minimum_precision": 0.5, "minimum_recall": 0.5},
    )
    assert response.status_code == 200
    assert response.json()["criterion"] == "max_f2"
