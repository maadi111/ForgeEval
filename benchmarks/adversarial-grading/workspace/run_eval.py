"""CLI runner to evaluate a model candidate using ModelEvaluator."""

import json
import pathlib
import sys

from evaluator import ModelEvaluator


def main():
    root = pathlib.Path(__file__).resolve().parent
    data_path = root / "data" / "probe_dataset.json"
    if not data_path.exists():
        # Fallback to shared benchmark data dir
        data_path = root.parent / "data" / "probe_dataset.json"

    if not data_path.exists():
        print(f"Error: probe dataset not found at {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    _evaluator = ModelEvaluator(dataset)
    print(f"Loaded evaluator with {len(dataset.get('records', []))} records.")
    print("Ready to evaluate model candidate.")


if __name__ == "__main__":
    main()
