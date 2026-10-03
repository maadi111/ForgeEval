# Benchmark: PyTorch & GPU Inference Latency Optimization

## Scenario Description

Our production real-time risk assessment service relies on a PyTorch deep neural network to score incoming transaction streams. Under live traffic spikes, the inference endpoint experiences severe latency degradation:

- A batch of 256 requests takes ~2.0 - 3.0 seconds to execute, dramatically exceeding our 99th percentile SLA threshold (target: < 0.25s for 256 items).
- High CPU/GPU utilization and excessive memory allocation under concurrent requests.
- Worker threads queue up and fail health checks under sustained load.

## Requirements

1. **Optimize `pipeline.py`**:
   - Accelerate `predict_batch()` to meet the throughput SLA (< 0.20s for 256 items).
   - Ensure the model inference pipeline is appropriately configured for production evaluation.
   - Maintain strict numerical correctness: predictions and probabilities must match reference outputs within numerical tolerance (`atol=1e-3`).
   - Guarantee batch invariance: `predict_batch([A, B])` must yield identical predictions to individual single-item predictions.
2. **Preserve Public API Contract**:
   - `InferencePipeline(weights_path)` must accept an optional weights file path.
   - `predict_one(features: list[float]) -> dict` returns `{"probability": float, "prediction": int}`.
   - `predict_batch(batch: list[dict]) -> list[dict]` returns a list of dictionaries with `{"request_id": str, "probability": float, "prediction": int}`.

## Public Test Verification

Run the public test suite to verify your baseline contract:

```bash
pytest ../tests/public/ -v
```
