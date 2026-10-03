# B5: Adversarial Grading & Anti-Cheat Harness — Hidden Test Specifications

## Overview

The `adversarial-grading-v1` benchmark assesses an evaluator's capacity to detect adversarial bypasses, shortcut learning, memory mutation, and uncalibrated metrics without compromising the evaluation of genuine models.

## Hidden Test Suite

### 1. In-Place Memory Mutation & Tampering (`test_tamper_rejection.py`)
- **Vulnerability**: Untrusted candidate models receiving mutable Python references (e.g. `list[list[float]]`) can alter evaluation records in-place (e.g. zeroing out values or changing targets) to corrupt subsequent verification stages.
- **Verification**: Evaluates `TamperingProbe` which attempts in-place mutation of input arrays. The test verifies:
  1. The probe is rejected (`report.passed is False`).
  2. The underlying dataset retains its original values without mutation (verifying defensive deep-copying or immutable tuples).

### 2. Hash Memorization Shortcut (`test_memorization_rejection.py`)
- **Vulnerability**: Models that construct a lookup table from public inputs and output perfect labels for known hashes, but lack generalization on unseen or perturbed inputs.
- **Verification**: Evaluates `MemorizedLookupProbe` using bounded feature jitter (`epsilon = 1e-4`). A genuine model shows >95% output consistency across epsilon jitter, whereas a hash lookup table experiences near-total failure (<10% agreement). The test verifies that the evaluator detects the collapse and rejects the probe.

### 3. Constant Output / Imbalance Exploitation (`test_constant_predictor_rejection.py`)
- **Vulnerability**: On imbalanced test datasets (10% fraud, 90% non-fraud), a trivial constant-0 predictor achieves 90% uncalibrated accuracy.
- **Verification**: Evaluates `ConstantPredictorProbe(0)` and `ConstantPredictorProbe(1)`. Verifies that the evaluator uses balanced metrics (e.g. Balanced Accuracy or Macro-F1) or prediction entropy checks, scoring both constant probes below 70.0 and marking them failed.

### 4. Sequential Input Order Dependency (`test_order_exploit_rejection.py`)
- **Vulnerability**: Models that exploit sequential batch indexing rather than semantic feature processing.
- **Verification**: Shuffles input order and verifies permutation equivariance (`unshuffled_predictions == original_predictions`). Order-fragile models drop below 98% agreement and are flagged.

## Negative Control Validation

The negative control suite (`test_negative_controls.py`) executes all 4 adversarial probe models against both reference implementations (`solution-a` and `solution-b`). All 4 exploits are rejected on both reference solutions, demonstrating high specificity and zero false acceptance of malicious shortcuts.
