from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.exceptions import DatasetValidationError

KNOWN_TARGET_COLUMNS = [
    "Class",
    "class",
    "isFraud",
    "Fraud",
    "fraud",
    "Is_Fraud",
    "is_fraud",
    "FLAG",
    "Label",
    "label",
    "target",
    "y",
    "LOAN_DEFAULT",
    "illicit",
    "Illicit",
]


@dataclass(slots=True)
class DatasetBundle:
    dataset_name: str
    path: Path
    frame: pd.DataFrame
    target_column: str
    feature_columns: list[str]


def standardize_target_column(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    for candidate in KNOWN_TARGET_COLUMNS:
        if candidate in frame.columns:
            renamed = frame.rename(columns={candidate: "target"})
            return renamed, candidate
    raise DatasetValidationError(
        "Nenhuma coluna de target conhecida foi encontrada no dataset."
    )


def load_dataset(dataset_path: str | Path, dataset_name: str | None = None) -> DatasetBundle:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em '{path}'. Adicione o arquivo ao workspace antes de executar o pipeline."
        )

    frame = pd.read_csv(path)
    if frame.empty:
        raise DatasetValidationError("O dataset carregado esta vazio.")

    standardized_frame, original_target = standardize_target_column(frame)
    standardized_frame = standardized_frame[standardized_frame["target"].isin([0, 1])].copy()
    if standardized_frame.empty:
        raise DatasetValidationError(
            "Nao ha amostras binarias validas apos padronizacao do target."
        )

    feature_columns = [column for column in standardized_frame.columns if column != "target"]
    if not feature_columns:
        raise DatasetValidationError("Nenhuma feature disponivel para treinamento.")

    return DatasetBundle(
        dataset_name=dataset_name or path.stem,
        path=path,
        frame=standardized_frame,
        target_column=original_target,
        feature_columns=feature_columns,
    )


def validate_feature_payload(payload: dict[str, object], expected_features: list[str]) -> None:
    received = set(payload.keys())
    expected = set(expected_features)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        raise DatasetValidationError(
            "Payload de features invalido. "
            f"Ausentes: {missing or 'nenhuma'} | Extras: {extra or 'nenhuma'}"
        )
