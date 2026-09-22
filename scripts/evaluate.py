from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.inference.artifact import load_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a trained model artifact.")
    parser.add_argument("--artifact", required=True, help="Path to a serialized artifact.")
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    print(json.dumps(
        {
            "model_name": artifact.model_name,
            "model_version": artifact.model_version,
            "dataset_name": artifact.dataset_name,
            "threshold": artifact.threshold,
            "metrics": artifact.metrics,
            "best_thresholds": artifact.best_thresholds,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
