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

    artifact_path = tmp_path / "trained.joblib"
    output = trainer.train_and_serialize(
        dataset_path=synthetic_dataset_csv,
        dataset_name="synthetic",
        artifact_path=artifact_path,
    )
    assert artifact_path.exists()
    assert output.artifact.model_name == "logistic_regression"
    assert not output.comparison_table.empty
