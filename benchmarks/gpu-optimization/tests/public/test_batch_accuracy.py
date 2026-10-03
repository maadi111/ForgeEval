"""Public test 2: Basic correctness and prediction accuracy."""

import json
import os
import pathlib
import sys

import pytest

WORKSPACE = pathlib.Path(os.environ.get("WORKSPACE", pathlib.Path(__file__).parents[2] / "workspace"))
sys.path.insert(0, str(WORKSPACE))

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
B4_DATA_DIR = REPO_ROOT / "benchmarks" / "gpu-optimization" / "data"


@pytest.fixture(scope="module")
def assets():
    weights_path = B4_DATA_DIR / "model_weights.pt"
    if not weights_path.exists():
        from generate_model import generate_benchmark_assets

        generate_benchmark_assets(B4_DATA_DIR)

    with open(B4_DATA_DIR / "eval_requests.json", "r", encoding="utf-8") as f:
        requests = json.load(f)["requests"][:16]

    with open(B4_DATA_DIR / "ground_truth.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    return {
        "weights": str(weights_path),
        "requests": requests,
        "gt_preds": gt["predictions"][:16],
    }


def test_batch_accuracy_sanity(assets):
    """Verify that predictions match ground truth expectations on visible subset."""
    from pipeline import InferencePipeline

    pipeline = InferencePipeline(assets["weights"])
    preds = pipeline.predict_batch(assets["requests"])

    assert len(preds) == len(assets["requests"])
    correct = sum(
        1 for p, y in zip(preds, assets["gt_preds"]) if p["prediction"] == y
    )
    accuracy = correct / len(assets["gt_preds"])
    assert accuracy >= 0.80, f"Baseline accuracy below expected threshold: {accuracy:.2f}"
