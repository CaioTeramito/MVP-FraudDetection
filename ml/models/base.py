from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class FraudModelStrategy(ABC):
    name: str

    @abstractmethod
    def train(self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_estimator(self):
        raise NotImplementedError
