# Grading Design

## Core rule

Never grade by checking whether a candidate file matches the reference solution. Define the input contract, output contract, correctness properties, invariants, resource constraints, regression requirements, and adversarial properties.

Bad:

```python
assert candidate_file == reference_file
```

Better:

```python
predictions = run_candidate(unseen_inputs)
assert accuracy(predictions, labels) >= 0.90
```

Stronger:

```python
assert temporal_integrity(candidate_dataset)
assert accuracy(predictions, labels) >= 0.90
assert latency_ms <= 100
assert handles_permuted_input(candidate)
assert handles_unseen_inputs(candidate)
```

## Negative controls

Every benchmark should include known-bad implementations:

- hardcoded output
- input memorization
- test-specific conditionals
- constant predictions
- output-order assumptions
- test tampering

## Score report

```python
{
    "passed": False,
    "score": 72.5,
    "components": {...},
    "failures": [{"test": "unseen_data", "reason": "accuracy below threshold"}],
}
```
