from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app.core.config import settings
from ml.data.dataset import DatasetBundle, load_dataset
from ml.evaluation.metrics import calculate_classification_metrics
from ml.inference.artifact import ModelArtifact, save_artifact
from ml.models.strategies import build_model_strategies
from ml.preprocessing.pipeline import build_preprocessing_pipeline, get_transformed_feature_names
from ml.threshold.analysis import ThresholdArtifacts, evaluate_thresholds


@dataclass(slots=True)
class TrainingOutput:
    artifact: ModelArtifact
    threshold_artifacts: ThresholdArtifacts
    comparison_table: pd.DataFrame
    test_frame: pd.DataFrame


def _plot_threshold_figures(
    threshold_artifacts: ThresholdArtifacts,
    results_dir: Path,
    model_name: str,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    table = threshold_artifacts.table

    for metric_name, output_name in [
        ("precision", "precision_threshold"),
        ("recall", "recall_threshold"),
        ("f1", "f1_threshold"),
        ("f2", "f2_threshold"),
        ("false_positive_rate", "fpr_threshold"),
        ("false_negative_rate", "fnr_threshold"),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(table["threshold"], table[metric_name], marker="o")
        plt.xlabel("Threshold")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"{metric_name.replace('_', ' ').title()} x Threshold - {model_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"{output_name}_{model_name}.png")
        plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(
        threshold_artifacts.precision_recall_curve["recall"],
        threshold_artifacts.precision_recall_curve["precision"],
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision x Recall - {model_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"precision_recall_{model_name}.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(
        threshold_artifacts.roc_curve["fpr"],
        threshold_artifacts.roc_curve["tpr"],
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"roc_curve_{model_name}.png")
    plt.close()


def train_and_serialize(
    dataset_path: str | Path,
    dataset_name: str | None = None,
    artifact_path: str | Path | None = None,
    model_version: str = "0.1.0",
    minimum_precision: float | None = None,
    minimum_recall: float | None = None,
    threshold_step: float | None = None,
    model_selection_metric: str = "cv_pr_auc_mean",
    threshold_selection_strategy: str = "max_f2",
) -> TrainingOutput:
    dataset = load_dataset(dataset_path, dataset_name=dataset_name)
    feature_frame = dataset.frame[dataset.feature_columns]
    target = dataset.frame["target"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        feature_frame,
        target,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=target,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=settings.validation_size,
        random_state=settings.random_state,
        stratify=y_train_full,
    )

    preprocessor, _ = build_preprocessing_pipeline(X_train)
    strategies = build_model_strategies(settings.random_state)
    cv = StratifiedKFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.random_state,
    )

    comparison_rows: list[dict[str, Any]] = []
    fitted_runs: list[dict[str, Any]] = []

    for strategy in strategies:
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("smote", SMOTE(random_state=settings.random_state)),
                ("classifier", strategy.get_estimator()),
            ]
        )
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        validation_probabilities = pipeline.predict_proba(X_val)[:, 1]
        validation_evaluation = calculate_classification_metrics(
            y_val.to_numpy(),
            validation_probabilities,
            threshold=0.5,
        )
        comparison_rows.append(
            {
                "model_name": strategy.name,
                "cv_pr_auc_mean": float(cv_scores.mean()),
                "cv_pr_auc_std": float(cv_scores.std()),
                "validation_accuracy": float(validation_evaluation.metrics["accuracy"]),
                "validation_precision": float(validation_evaluation.metrics["precision"]),
                "validation_recall": float(validation_evaluation.metrics["recall"]),
                "validation_f1": float(validation_evaluation.metrics["f1"]),
                "validation_f2": float(validation_evaluation.metrics["f2"]),
                "validation_roc_auc": float(validation_evaluation.metrics["roc_auc"]),
                "validation_pr_auc": float(validation_evaluation.metrics["pr_auc"]),
            }
        )
        fitted_runs.append(
            {
                "model_name": strategy.name,
                "pipeline": pipeline,
                "validation_probabilities": validation_probabilities,
                "validation_evaluation": validation_evaluation,
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)
    ranking_columns = [model_selection_metric, "validation_f2", "validation_recall"]
    comparison_table = comparison_table.sort_values(ranking_columns, ascending=[False, False, False])

    selected_run = fitted_runs[0]
    for run in fitted_runs:
        if run["model_name"] == comparison_table.iloc[0]["model_name"]:
            selected_run = run
            break

    threshold_artifacts = evaluate_thresholds(
        y_true=y_val.to_numpy(),
        probabilities=selected_run["validation_probabilities"],
        step=threshold_step or settings.threshold_step,
        minimum_precision=minimum_precision or settings.minimum_precision,
        minimum_recall=minimum_recall or settings.minimum_recall,
    )

    if threshold_selection_strategy not in threshold_artifacts.best_thresholds:
        available = ", ".join(sorted(threshold_artifacts.best_thresholds))
        raise ValueError(
            f"Threshold strategy '{threshold_selection_strategy}' invalida. Disponiveis: {available}"
        )

    selected_threshold = threshold_artifacts.best_thresholds[threshold_selection_strategy]["threshold"]
    final_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("smote", SMOTE(random_state=settings.random_state)),
            ("classifier", clone(selected_run["pipeline"].named_steps["classifier"])),
        ]
    )
    final_pipeline.fit(X_train_full, y_train_full)
    final_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = calculate_classification_metrics(
        y_true=y_test.to_numpy(),
        probabilities=final_probabilities,
        threshold=selected_threshold,
    )

    fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
    estimator = final_pipeline.named_steps["classifier"]
    transformed_feature_names = get_transformed_feature_names(fitted_preprocessor)

    artifact = ModelArtifact.create(
        model_name=selected_run["model_name"],
        model_version=model_version,
        dataset_name=dataset.dataset_name,
        dataset_path=str(dataset.path),
        target_column=dataset.target_column,
        feature_columns=dataset.feature_columns,
        transformed_feature_names=transformed_feature_names,
        threshold=selected_threshold,
        threshold_low=settings.threshold_low,
        threshold_high=settings.threshold_high,
        metrics=final_metrics.metrics,
        threshold_table=threshold_artifacts.table,
        best_thresholds=threshold_artifacts.best_thresholds,
        holdout_truth=y_test.astype(int).tolist(),
        holdout_probabilities=[float(value) for value in final_probabilities],
        training_config={
            "test_size": settings.test_size,
            "validation_size": settings.validation_size,
            "cv_folds": settings.cv_folds,
            "random_state": settings.random_state,
            "threshold_step": threshold_step or settings.threshold_step,
            "minimum_precision": minimum_precision or settings.minimum_precision,
            "minimum_recall": minimum_recall or settings.minimum_recall,
            "model_candidates": [strategy.name for strategy in strategies],
            "model_selection_metric": model_selection_metric,
            "threshold_selection_strategy": threshold_selection_strategy,
        },
        preprocessor=fitted_preprocessor,
        estimator=estimator,
        shap_available=settings.enable_shap,
        notes=[
            "O modelo serializado do MVP e escolhido pela metrica configurada em "
            f"'model_selection_metric={model_selection_metric}' sobre treino/validacao.",
            f"O threshold operacional foi escolhido pela estrategia '{threshold_selection_strategy}' sobre o conjunto de validacao.",
            "O conjunto de teste final foi preservado para avaliacao final e nao participa da selecao.",
            "A comparacao completa entre candidatos permanece salva em results/metrics/model_comparison.csv.",
        ],
    )

    save_artifact(artifact, artifact_path or settings.model_artifact_path)

    results_dir = settings.results_dir
    threshold_artifacts.table.to_csv(
        results_dir / "threshold" / f"{artifact.model_name}_thresholds.csv",
        index=False,
    )
    comparison_table.to_csv(
        results_dir / "metrics" / "model_comparison.csv",
        index=False,
    )
    _plot_threshold_figures(
        threshold_artifacts,
        results_dir / "figures",
        artifact.model_name,
    )

    test_frame = X_test.copy()
    test_frame["target"] = y_test
    test_frame["fraud_probability"] = final_probabilities

    return TrainingOutput(
        artifact=artifact,
        threshold_artifacts=threshold_artifacts,
        comparison_table=comparison_table,
        test_frame=test_frame,
    )
