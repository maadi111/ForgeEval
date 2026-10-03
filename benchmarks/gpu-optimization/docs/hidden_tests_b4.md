# B4: GPU & PyTorch Inference Optimization — Hidden Test Specifications

## Overview

The `gpu-optimization-v1` benchmark evaluates a candidate's capacity to profile, diagnose, and optimize a deep neural network inference pipeline suffering from severe latency degradation and autograd overhead under strict production SLA.

## Hidden Test Suite

### 1. Throughput & Latency SLA (`test_inference_latency_sla.py`)
- **Vulnerability**: The naive implementation processes input items sequentially in a Python `for` loop, repeatedly dispatching single-row tensors across the Python/C++ interface with active autograd tracking. Under a batch of 256 transactions, processing time takes ~2.0 - 3.0 seconds.
- **Verification**: Times execution of `predict_batch()` on 256 items. Requires total elapsed time < 0.35s (SLA threshold). Optimized implementations with vectorized micro-batching and `torch.inference_mode()` achieve < 0.10s (>2,500 samples/sec), comfortably passing.

### 2. Batch Invariance (`test_batch_invariance.py`)
- **Vulnerability**: Implementations that alter sample order or introduce inter-sample dependencies during batching (e.g. uncalibrated BatchNorm or sorting without inverse permutation).
- **Verification**: Evaluates `predict_batch([A, B, C])` against `[predict_one(A), predict_one(B), predict_one(C)]`. Requires 100% agreement on class predictions and probability difference < 1e-3.

### 3. Numerical Precision & Ground Truth (`test_numerical_precision.py`)
- **Vulnerability**: Implementations that take numerical shortcuts (e.g. extreme quantization, skipping residual connections, or returning coarse approximations).
- **Verification**: Compares all 256 predictions and probabilities against pre-computed ground truth with exact discrete match and maximum probability difference < 1e-3.

## Negative Control Validation

The negative control suite (`test_negative_controls.py`) verifies:
1. Constant Predictor Shortcut: always outputting class 0 achieves low accuracy (<70%) and fails.
2. Truncated Batch Shortcut: returning fewer items than the batch input fails length validation.
3. Reference Solutions: both `solution-a` (vectorized batching + `inference_mode`) and `solution-b` (TorchScript JIT tracing + chunked batching) satisfy all correctness and latency criteria.
