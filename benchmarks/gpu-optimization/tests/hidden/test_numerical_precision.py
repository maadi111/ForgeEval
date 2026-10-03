"""Hidden test 3: Numerical precision and ground truth alignment across full evaluation set."""

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
def benchmark_data():
    weights_path = B4_DATA_DIR / "model_weights.pt"
    if not weights_path.exists():
        from generate_model import generate_benchmark_assets

        generate_benchmark_assets(B4_DATA_DIR)

    with open(B4_DATA_DIR / "eval_requests.json", "r", encoding="utf-8") as f:
        requests = json.load(f)["requests"]

    with open(B4_DATA_DIR / "ground_truth.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    return {
        "weights": str(weights_path),
        "requests": requests,
        "gt_preds": gt["predictions"],
        "gt_probs": gt["probabilities"],
    }


def test_numerical_precision(benchmark_data):
    """Verify that predictions match ground truth with exact discrete agreement and tight probability bounds."""
    from pipeline import InferencePipeline

    pipeline = InferencePipeline(benchmark_data["weights"])
    preds = pipeline.predict_batch(benchmark_data["requests"])

    assert len(preds) == 256
    exact_match = sum(
        1 for p, y in zip(preds, benchmark_data["gt_preds"]) if p["prediction"] == y
    )
    accuracy = exact_match / 256
    assert accuracy == 1.0, f"Expected 100% agreement with ground truth, got {accuracy:.2%}"

    prob_diffs = [
        abs(p["probability"] - y)
        for p, y in zip(preds, benchmark_data["gt_probs"])
    ]
    max_diff = max(prob_diffs)
    assert max_diff < 1e-3, f"Max probability deviation too high: {max_diff:.6f}"
