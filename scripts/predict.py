from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.inference.predictor import FraudPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local fraud prediction from a JSON payload.")
    parser.add_argument("--artifact", required=True, help="Serialized model artifact path.")
    parser.add_argument("--input", required=True, help="Path to a JSON file with feature values.")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    predictor = FraudPredictor.from_path(args.artifact)
    result = predictor.predict_transaction(payload)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
