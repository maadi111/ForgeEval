"""Negative controls for B4: Verify rejection of shortcuts and acceptance of reference solutions."""

import json
import pathlib
import sys

import pytest

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
    }


class ConstantPredictorPipeline:
    """Negative control: trivial constant prediction shortcut."""

    def __init__(self, *args, **kwargs):
        pass

    def predict_one(self, features):
        return {"probability": 0.0, "prediction": 0}

    def predict_batch(self, batch):
        return [
            {"request_id": r.get("request_id", ""), "probability": 0.0, "prediction": 0}
            for r in batch
        ]


class TruncatedBatchPipeline:
    """Negative control: truncated batch prediction shortcut."""

    def __init__(self, *args, **kwargs):
        pass

    def predict_one(self, features):
        return {"probability": 0.5, "prediction": 0}

    def predict_batch(self, batch):
        # Only returns first 16 items
        return [
            {"request_id": r.get("request_id", ""), "probability": 0.5, "prediction": 0}
            for r in batch[:16]
        ]


def test_negative_control_constant_predictor_fails(benchmark_data):
    """Negative control: Constant predictor must fail accuracy check."""
    pipeline = ConstantPredictorPipeline()
    preds = pipeline.predict_batch(benchmark_data["requests"])
    correct = sum(
        1 for p, y in zip(preds, benchmark_data["gt_preds"]) if p["prediction"] == y
    )
    accuracy = correct / len(benchmark_data["gt_preds"])
    assert accuracy < 0.70, f"Constant predictor scored unexpectedly high: {accuracy:.2%}"


def test_negative_control_truncated_batch_fails(benchmark_data):
    """Negative control: Truncated batch must fail length check."""
    pipeline = TruncatedBatchPipeline()
    preds = pipeline.predict_batch(benchmark_data["requests"])
    assert len(preds) != len(benchmark_data["requests"])


def _load_pipeline_cls(sol_name: str):
    sol_dir = REPO_ROOT / "benchmarks" / "gpu-optimization" / "reference" / sol_name
    for mod in ["pipeline", "model"]:
        if mod in sys.modules:
            del sys.modules[mod]
    sys.path.insert(0, str(sol_dir))
    try:
        import pipeline

        return pipeline.InferencePipeline
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("sol_name", ["solution-a", "solution-b"])
def test_reference_solutions_pass_negative_controls(benchmark_data, sol_name):
    """Reference solutions must satisfy accuracy and contract."""
    cls = _load_pipeline_cls(sol_name)
    pipeline = cls(benchmark_data["weights"])
    preds = pipeline.predict_batch(benchmark_data["requests"])
    assert len(preds) == 256
    correct = sum(
        1 for p, y in zip(preds, benchmark_data["gt_preds"]) if p["prediction"] == y
    )
    accuracy = correct / 256
    assert accuracy == 1.0
