from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(slots=True)
class PreprocessingArtifacts:
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_high_cardinality_features: dict[str, int]


def infer_feature_types(
    frame: pd.DataFrame,
    max_categorical_cardinality: int,
) -> PreprocessingArtifacts:
    numeric_features = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features: list[str] = []
    dropped_high_cardinality_features: dict[str, int] = {}

    for column in frame.columns:
        if column in numeric_features:
            continue
        cardinality = int(frame[column].nunique(dropna=True))
        if cardinality > max_categorical_cardinality:
            dropped_high_cardinality_features[column] = cardinality
        else:
            categorical_features.append(column)

    return PreprocessingArtifacts(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_high_cardinality_features=dropped_high_cardinality_features,
    )


def build_preprocessing_pipeline(
    frame: pd.DataFrame,
    max_categorical_cardinality: int = 1000,
) -> tuple[ColumnTransformer, PreprocessingArtifacts]:
    """Build preprocessing without one-hot encoding high-cardinality identifiers.

    Identifiers such as account numbers are unsuitable for dense one-hot encoding:
    they create a mostly empty matrix that can exceed available memory by orders
    of magnitude. They are excluded based on training-partition cardinality.
    """
    feature_types = infer_feature_types(frame, max_categorical_cardinality)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, feature_types.numeric_features),
            ("cat", categorical_pipeline, feature_types.categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, feature_types


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return preprocessor.get_feature_names_out().tolist()
