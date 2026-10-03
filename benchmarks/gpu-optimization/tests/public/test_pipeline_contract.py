"""Public test 1: Verify InferencePipeline API contract and output schemas."""

import os
import pathlib
import sys

import pytest

WORKSPACE = pathlib.Path(os.environ.get("WORKSPACE", pathlib.Path(__file__).parents[2] / "workspace"))
sys.path.insert(0, str(WORKSPACE))

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
B4_DATA_DIR = REPO_ROOT / "benchmarks" / "gpu-optimization" / "data"


@pytest.fixture(scope="module")
def weights_path():
    path = B4_DATA_DIR / "model_weights.pt"
    if not path.exists():
        from generate_model import generate_benchmark_assets

        generate_benchmark_assets(B4_DATA_DIR)
    return str(path)


def test_pipeline_contract(weights_path):
    """Verify that InferencePipeline conforms to public method signatures and schema."""
    from pipeline import InferencePipeline

    pipeline = InferencePipeline(weights_path)
    assert hasattr(pipeline, "predict_one")
    assert hasattr(pipeline, "predict_batch")

    # Test single prediction
    dummy_feats = [0.1] * 64
    res_one = pipeline.predict_one(dummy_feats)
    assert isinstance(res_one, dict)
    assert "probability" in res_one
    assert "prediction" in res_one
    assert 0.0 <= res_one["probability"] <= 1.0
    assert res_one["prediction"] in (0, 1)

    # Test batch prediction
    batch = [
        {"request_id": "r1", "features": [0.1] * 64},
        {"request_id": "r2", "features": [-0.2] * 64},
    ]
    res_batch = pipeline.predict_batch(batch)
    assert isinstance(res_batch, list)
    assert len(res_batch) == 2
    assert res_batch[0]["request_id"] == "r1"
    assert res_batch[1]["request_id"] == "r2"
