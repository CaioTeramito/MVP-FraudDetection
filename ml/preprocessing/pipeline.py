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


def infer_feature_types(frame: pd.DataFrame) -> PreprocessingArtifacts:
    numeric_features = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in frame.columns if column not in numeric_features
    ]
    return PreprocessingArtifacts(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def build_preprocessing_pipeline(frame: pd.DataFrame) -> tuple[ColumnTransformer, PreprocessingArtifacts]:
    feature_types = infer_feature_types(frame)

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
