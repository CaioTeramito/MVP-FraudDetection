from __future__ import annotations

import pytest

from app.core.exceptions import DatasetValidationError
from ml.data.dataset import load_dataset, validate_feature_payload


def test_load_dataset_standardizes_target(synthetic_dataset_csv):
    bundle = load_dataset(synthetic_dataset_csv, dataset_name="synthetic")
    assert bundle.dataset_name == "synthetic"
    assert bundle.target_column == "Class"
    assert "target" in bundle.frame.columns
    assert len(bundle.feature_columns) == 6


def test_validate_feature_payload_detects_missing_features():
    with pytest.raises(DatasetValidationError):
        validate_feature_payload({"feature_1": 1.0}, ["feature_1", "feature_2"])
