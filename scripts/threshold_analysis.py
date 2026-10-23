from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from ml.inference.artifact import load_artifact
from ml.threshold.analysis import evaluate_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute threshold analysis from an artifact.")
    parser.add_argument("--artifact", required=True, help="Serialized model artifact path.")
    parser.add_argument("--step", type=float, default=settings.threshold_step)
    parser.add_argument("--minimum-precision", type=float, default=settings.minimum_precision)
    parser.add_argument("--minimum-recall", type=float, default=settings.minimum_recall)
    parser.add_argument("--model", default=None, help="Candidate model name stored in the artifact.")
    parser.add_argument("--cost-fn", type=float, default=None)
    parser.add_argument("--cost-fp", type=float, default=None)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    selected = artifact.model_results.get(args.model) if args.model else None
    selected = selected or {
        "model_name": artifact.model_name,
        "validation_truth": artifact.holdout_truth,
        "validation_probabilities": artifact.holdout_probabilities,
    }
    threshold_artifacts = evaluate_thresholds(
        y_true=pd.Series(selected["validation_truth"]).to_numpy(),
        probabilities=pd.Series(selected["validation_probabilities"]).to_numpy(),
        step=args.step,
        minimum_precision=args.minimum_precision,
        minimum_recall=args.minimum_recall,
        strategy=(
            __import__("ml.threshold.strategies", fromlist=["build_threshold_strategy"])
            .build_threshold_strategy("cost", cost_fn=args.cost_fn, cost_fp=args.cost_fp)
            if args.cost_fn is not None and args.cost_fp is not None else None
        ),
    )
    model_name = selected["model_name"]
    output_path = settings.results_dir / "threshold" / f"{model_name}_thresholds_recomputed.csv"
    threshold_artifacts.table.to_csv(output_path, index=False)
    print(threshold_artifacts.table.to_string(index=False))
    print(f"Analise salva em: {output_path}")
    print(f"Melhores thresholds: {threshold_artifacts.best_thresholds}")


if __name__ == "__main__":
    main()
