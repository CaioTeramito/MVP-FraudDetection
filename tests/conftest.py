from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from app.core.config import settings
from app.services.model_service import get_predictor
from ml.inference.artifact import ModelArtifact, save_artifact
from ml.preprocessing.pipeline import build_preprocessing_pipeline, get_transformed_feature_names
from ml.threshold.analysis import evaluate_thresholds


@pytest.fixture
def synthetic_dataset_csv(tmp_path: Path) -> Path:
    features, target = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.92, 0.08],
        random_state=42,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(6)])
    frame["Class"] = target
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_artifact(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 1.0, 1.2, 0.4, 0.9],
            "feature_2": [1, 0, 1, 0, 1, 0],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    X = frame[["feature_1", "feature_2"]]
    y = frame["target"]
    preprocessor, _ = build_preprocessing_pipeline(X)
    X_transformed = preprocessor.fit_transform(X)
    estimator = LogisticRegression(max_iter=500, random_state=42)
    estimator.fit(X_transformed, y)
    probabilities = estimator.predict_proba(X_transformed)[:, 1]
    threshold_artifacts = evaluate_thresholds(
        y_true=y.to_numpy(),
        probabilities=probabilities,
        step=0.05,
        minimum_precision=0.5,
        minimum_recall=0.5,
    )
    artifact = ModelArtifact.create(
        model_name="logistic_regression",
        model_version="test",
        dataset_name="synthetic",
        dataset_path="synthetic.csv",
        target_column="Class",
        feature_columns=["feature_1", "feature_2"],
        transformed_feature_names=get_transformed_feature_names(preprocessor),
        threshold=threshold_artifacts.best_thresholds["max_f2"]["threshold"],
        threshold_low=0.3,
        threshold_high=0.7,
        metrics=threshold_artifacts.table.iloc[0].to_dict(),
        threshold_table=threshold_artifacts.table,
        best_thresholds=threshold_artifacts.best_thresholds,
        holdout_truth=y.astype(int).tolist(),
        holdout_probabilities=[float(value) for value in probabilities],
        training_config={"source": "pytest"},
        preprocessor=preprocessor,
        estimator=estimator,
        shap_available=False,
    )
    artifact_path = tmp_path / "artifact.joblib"
    save_artifact(artifact, artifact_path)
    return artifact_path


@pytest.fixture(autouse=True)
def isolate_model_artifact(monkeypatch: pytest.MonkeyPatch, sample_artifact: Path) -> None:
    monkeypatch.setattr(settings, "model_artifact_path", sample_artifact)
    get_predictor.cache_clear()
