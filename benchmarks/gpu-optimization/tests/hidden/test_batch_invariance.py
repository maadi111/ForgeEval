"""Hidden test 2: Batch invariance — batch processing must yield identical results to single prediction."""

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
        requests = json.load(f)["requests"][:32]

    return {"weights": str(weights_path), "requests": requests}


def test_batch_invariance(benchmark_data):
    """Verify f([A, B]) == [f(A), f(B)] within numerical tolerance."""
    from pipeline import InferencePipeline

    pipeline = InferencePipeline(benchmark_data["weights"])
    requests = benchmark_data["requests"]

    # 1. Evaluate single predictions
    single_preds = []
    for req in requests:
        res = pipeline.predict_one(req["features"])
        single_preds.append(res)

    # 2. Evaluate batch predictions
    batch_preds = pipeline.predict_batch(requests)

    assert len(single_preds) == len(batch_preds)
    for i, (s, b) in enumerate(zip(single_preds, batch_preds)):
        assert s["prediction"] == b["prediction"], f"Prediction mismatch at index {i}: single={s}, batch={b}"
        diff = abs(s["probability"] - b["probability"])
        assert diff < 1e-3, f"Probability drift exceeded tolerance at index {i}: diff={diff:.6f}"
