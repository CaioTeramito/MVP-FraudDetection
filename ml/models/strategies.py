from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

from ml.models.base import FraudModelStrategy


class _BaseSklearnStrategy(FraudModelStrategy):
    def __init__(self, estimator, name: str) -> None:
        self.estimator = estimator
        self.name = name

    def train(self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray) -> None:
        self.estimator.fit(features, target)

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(features)[:, 1]

    def get_estimator(self):
        return self.estimator


def build_model_strategies(random_state: int) -> list[FraudModelStrategy]:
    strategies: list[FraudModelStrategy] = [
        _BaseSklearnStrategy(
            LogisticRegression(max_iter=1000, random_state=random_state),
            "logistic_regression",
        ),
        _BaseSklearnStrategy(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=1,
                class_weight=None,
            ),
            "random_forest",
        ),
    ]
    if XGBClassifier is not None:
        strategies.append(
            _BaseSklearnStrategy(
                XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    n_jobs=1,
                ),
                "xgboost",
            ),
        )
    return strategies
