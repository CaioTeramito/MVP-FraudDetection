from __future__ import annotations

import pandas as pd

from ml.preprocessing.pipeline import infer_feature_types


def test_high_cardinality_categorical_features_are_excluded():
    frame = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0],
            "type": ["PAYMENT", "TRANSFER", "PAYMENT"],
            "account_id": ["A1", "A2", "A3"],
        }
    )

    artifacts = infer_feature_types(frame, max_categorical_cardinality=2)

    assert artifacts.numeric_features == ["amount"]
    assert artifacts.categorical_features == ["type"]
    assert artifacts.dropped_high_cardinality_features == {"account_id": 3}
