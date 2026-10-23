from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from ml.models.strategies import _BaseSklearnStrategy
from ml.training import trainer


def test_train_and_serialize_runs_on_small_dataset(monkeypatch, synthetic_dataset_csv, tmp_path):
    monkeypatch.setattr(trainer.settings, "cv_folds", 3)
    monkeypatch.setattr(trainer.settings, "test_size", 0.25)
    monkeypatch.setattr(
        trainer,
        "build_model_strategies",
        lambda random_state: [
            _BaseSklearnStrategy(
                LogisticRegression(max_iter=500, random_state=random_state),
                "logistic_regression",
            )
        ],
    )
    captured = {}
    original_evaluate_thresholds = trainer.evaluate_thresholds

    def capture_threshold_input(*args, **kwargs):
        captured["y_true"] = kwargs["y_true"].copy()
        return original_evaluate_thresholds(*args, **kwargs)

    monkeypatch.setattr(trainer, "evaluate_thresholds", capture_threshold_input)

    artifact_path = tmp_path / "trained.joblib"
    output = trainer.train_and_serialize(
        dataset_path=synthetic_dataset_csv,
        dataset_name="synthetic",
        artifact_path=artifact_path,
    )
    assert artifact_path.exists()
    assert output.artifact.model_name == "logistic_regression"
    assert not output.comparison_table.empty
    assert "y_true" in captured
    assert captured["y_true"].tolist() != output.artifact.holdout_truth


def test_training_persists_validation_results_for_every_candidate(
    monkeypatch, synthetic_dataset_csv, tmp_path
):
    monkeypatch.setattr(trainer.settings, "cv_folds", 3)
    monkeypatch.setattr(trainer.settings, "test_size", 0.25)
    monkeypatch.setattr(
        trainer,
        "build_model_strategies",
        lambda random_state: [
            _BaseSklearnStrategy(
                LogisticRegression(max_iter=500, random_state=random_state), name
            )
            for name in ("random_forest", "logistic_regression", "xgboost")
        ],
    )

    artifact_path = tmp_path / "trained.joblib"
    output = trainer.train_and_serialize(
        dataset_path=synthetic_dataset_csv,
        dataset_name="synthetic",
        artifact_path=artifact_path,
    )

    assert set(output.artifact.model_results) == {
        "random_forest",
        "logistic_regression",
        "xgboost",
    }
    for model_name, result in output.artifact.model_results.items():
        assert result["model_name"] == model_name
        assert len(result["validation_truth"]) == len(result["validation_probabilities"])
        assert result["threshold_table"]
        assert "validation_metrics" in result

    from ml.inference.artifact import load_artifact
    loaded = load_artifact(artifact_path)
    assert set(loaded.model_results) == set(output.artifact.model_results)
