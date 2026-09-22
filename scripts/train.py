from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from ml.training.trainer import train_and_serialize


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and serialize the fraud model.")
    parser.add_argument("--dataset", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--dataset-name", default=None, help="Logical dataset name.")
    parser.add_argument(
        "--artifact-path",
        default=str(settings.model_artifact_path),
        help="Destination path for the serialized artifact.",
    )
    parser.add_argument(
        "--model-selection-metric",
        default="cv_pr_auc_mean",
        choices=[
            "cv_pr_auc_mean",
            "validation_pr_auc",
            "validation_f2",
            "validation_recall",
            "validation_precision",
        ],
    )
    parser.add_argument(
        "--threshold-selection-strategy",
        default="max_f2",
        choices=[
            "max_f1",
            "max_f2",
            "max_recall_with_min_precision",
            "max_precision_with_min_recall",
        ],
    )
    args = parser.parse_args()

    output = train_and_serialize(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        artifact_path=args.artifact_path,
        model_selection_metric=args.model_selection_metric,
        threshold_selection_strategy=args.threshold_selection_strategy,
    )
    print("Treinamento concluido.")
    print(output.comparison_table.to_string(index=False))
    print(f"Modelo serializado: {args.artifact_path}")
    print(f"Threshold selecionado: {output.artifact.threshold:.2f}")


if __name__ == "__main__":
    main()
