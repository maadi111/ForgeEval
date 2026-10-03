# Model Serving Benchmark

## Candidate Symptom

Our production scoring service passes individual unit tests and health checks, yet customer complaints indicate that during peak traffic, users occasionally receive predictions that do not correspond to their submitted attributes. When tested in isolation with single requests, the model appears accurate, but error rates spike under batched and concurrent traffic.

## Candidate Objective

Diagnose and repair the inference serving pipeline so that predictions are strictly invariant to batching, request ordering, and client serialization order, while satisfying latency and correctness guarantees.

## Planned Evaluation Criteria

- Correctness on standard benchmark samples
- Batch size and ordering invariance ($f([A, B]) == [f(A), f(B)]$)
- Payload key-order permutation invariance
- Concurrency and throughput stability
- Latency threshold compliance (< 25ms per item under batching)

The reference implementation is not the only accepted solution.
