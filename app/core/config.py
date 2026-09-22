from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_name: str = os.getenv("APP_NAME", "Fraud Detection MVP")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    model_artifact_path: Path = BASE_DIR / os.getenv(
        "MODEL_ARTIFACT_PATH",
        "models/fraud_model_artifact.joblib",
    )
    default_dataset_path: Path = BASE_DIR / os.getenv(
        "DEFAULT_DATASET_PATH",
        "data/raw/creditcard.csv",
    )
    results_dir: Path = BASE_DIR / os.getenv("RESULTS_DIR", "results")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    validation_size: float = float(os.getenv("VALIDATION_SIZE", "0.25"))
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))
    threshold_step: float = float(os.getenv("THRESHOLD_STEP", "0.05"))
    minimum_precision: float = float(os.getenv("MINIMUM_PRECISION", "0.50"))
    minimum_recall: float = float(os.getenv("MINIMUM_RECALL", "0.70"))
    threshold_low: float = float(os.getenv("THRESHOLD_LOW", "0.30"))
    threshold_high: float = float(os.getenv("THRESHOLD_HIGH", "0.70"))
    enable_shap: bool = os.getenv("ENABLE_SHAP", "true").lower() == "true"
    cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
            if origin.strip()
        ]
    )


settings = Settings()
