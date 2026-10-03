# Hidden Test Rationale — B1 Feature Leakage

This document explains why each hidden test exists and what candidate
behaviour it is designed to detect.

---

## `test_temporal_integrity.py`

### `test_sort_invariant`

**Why it exists:**  
A correct point-in-time implementation sorts the input by date before computing
rolling statistics.  A candidate might accidentally depend on row order: if the
input is already sorted, the leaky `rolling(7).mean()` and the correct
`shift(1).rolling(7).mean()` can produce the same first few values, masking the
bug.  Shuffling the input forces the implementation to sort explicitly.

**What it catches:**  
Any pipeline that assumes the input is sorted, or that computes features based
on positional row indices rather than time-ordered values.

---

### `test_future_row_perturbation_does_not_affect_past_features`

**Why it exists:**  
This is the definitive test of point-in-time correctness.  In a leaky pipeline,
`rolling_mean_7d` at date T includes `sales[T]`.  If we massively inflate
`sales[T]`, that value "bleeds backward" into the rolling window of the 6
preceding dates.  A correct pipeline prevents this: the feature at any past
date must be immutable once computed.

**What it catches:**  
- `rolling(7).mean()` without a shift (direct leakage)
- Any feature that computes a running statistic in the forward direction

---

### `test_feature_is_identical_with_or_without_subsequent_rows`

**Why it exists:**  
A real serving system computes features for a single timestamp with no
knowledge of what rows come after it.  If the feature value at date T changes
when the batch happens to include T+1, T+2, …, the implementation is leaky in
a production-unsafe way even if the window is formally "past-looking" on
sorted data.

**What it catches:**  
Implementations where the rolling window size or boundary shifts when the
dataset grows.

---

## `test_unseen_timestamps.py`

### `test_model_accuracy_on_holdout`

**Why it exists:**  
The leaky pipeline inflates in-sample validation metrics.  A model trained on
leaky features may appear to achieve very low MAE on a cross-validation split
taken from the same time range, but fails on genuinely held-out future data
because the leaky features are impossible to compute at inference time.

**What it catches:**  
The symptom: validation MAE looks good but production MAE collapses.
The 20.0 MAE threshold is calibrated so that a model trained on a fixed
pipeline passes, while a model trained on the leaky pipeline fails.

---

### `test_features_work_on_adversarial_timestamps`

**Why it exists:**  
Candidates might hard-code date ranges (e.g., checking `year <= 2024` in a
feature branch).  The adversarial data uses seed=99 and dates from 2025,
ensuring the pipeline is stateless with respect to the calendar.

**What it catches:**  
Hard-coded date assumptions, calendar-based conditionals, or any logic that
special-cases the training period.

---

## `test_negative_controls.py`

### `test_nc1_constant_predictor_fails_mae_threshold`

**Why it exists:**  
Validates that the MAE threshold is tight enough to reject a trivially bad
predictor.  If a constant prediction of the training mean passes, the threshold
is too lenient.

**What it catches:**  
Regressions in the grading threshold calibration.

---

### `test_nc2_memorised_output_fails_on_adversarial_seed`

**Why it exists:**  
Guards against a candidate who replays memorised training labels as predictions.
Because the adversarial holdout uses a different random seed and date range,
memorised training values will not match and must produce high MAE.

**What it catches:**  
Input-memorisation shortcuts; ensures hidden data is genuinely unseen.

---

### `test_nc3_leaky_pipeline_fails_temporal_integrity`

**Why it exists:**  
A meta-test that verifies the temporal integrity check itself is effective.
If this test fails, it means the perturbation magnitude or cut-off selection
is miscalibrated and the hidden test could be gamed by a leaky pipeline that
happens to pass due to numerical coincidence.

**What it catches:**  
Defects in the grading harness (self-validating test design).
