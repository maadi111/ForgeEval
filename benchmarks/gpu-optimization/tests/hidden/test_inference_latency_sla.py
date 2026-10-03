"""Hidden test 1: Inference latency must satisfy production SLA (< 0.35s for 256 items)."""

import json
import os
import pathlib
import sys
import time

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

    return {"weights": str(weights_path), "requests": requests}


def test_inference_latency_sla(benchmark_data):
    """Verify that batch inference satisfies SLA (< 0.35s for 256 samples)."""
    from pipeline import InferencePipeline

    pipeline = InferencePipeline(benchmark_data["weights"])
    requests = benchmark_data["requests"]
    assert len(requests) == 256

    # Warm-up pass
    _ = pipeline.predict_batch(requests[:16])

    # Timed benchmark pass
    t0 = time.perf_counter()
    preds = pipeline.predict_batch(requests)
    elapsed = time.perf_counter() - t0

    assert len(preds) == 256
    assert elapsed < 0.08, f"Inference latency SLA breached: took {elapsed:.3f}s for 256 items (target < 0.08s)"
