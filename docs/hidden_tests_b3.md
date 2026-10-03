# Hidden Test Rationale — B3 Broken Model Serving

This document details why each hidden test and adversarial check exists in the Model Serving benchmark, and what production failure mode it detects.

---

## 1. `test_batch_invariance.py`

### `test_batch_matches_single_predictions`

**Why it exists:**  
In high-throughput serving systems (e.g. Triton, TorchServe, vLLM, custom micro-batchers), requests are collected into batches to saturate hardware. A frequent bug is sorting batch items by payload size, length, or priority to reduce tensor padding or cache lookups, but failing to invert the sorting permutation before returning responses to the HTTP/gRPC handlers.

**What it catches:**  
- Unsorted / misaligned batch responses.
- Dynamic batchers that leak responses across request boundaries.
- Preprocessors that share state between items in a batch.

---

## 2. `test_key_permutation_invariance.py`

### `test_reversed_key_order` & `test_alphabetical_key_order`

**Why it exists:**  
RFC 8259 states that JSON object keys are inherently unordered collections of key/value pairs. Many client SDKs, proxy middleware, or serialization libraries (e.g., `orjson`, Python `json`, Java Jackson, Go `encoding/json`) emit keys in different orders. A serving preprocessor that parses dictionary values using `.values()` or iteration over `dict.keys()` assumes dictionary key order equals feature vector order.

**What it catches:**  
- Silent feature transposition bugs caused by dictionary key iteration.
- Preprocessing pipelines without schema validation.

---

## 3. `test_negative_controls.py`

### `test_nc1_constant_predictor_fails_discrimination`

**Why it exists:**  
Guarantees that degenerate or constant predictors (e.g., hardcoding the dominant class or returning fixed 0.5 probability) fail the evaluation threshold.

### `test_nc2_unsorted_batching_fails_invariance`

**Why it exists:**  
Validates that our evaluation suite itself reliably detects batch sorting errors, preventing false negatives.

### `test_nc3_key_order_dependent_parser_fails`

**Why it exists:**  
Verifies that key-order dependency is properly isolated and detected by our test harness.
