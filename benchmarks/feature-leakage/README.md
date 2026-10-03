# Feature Leakage Benchmark

## Candidate symptom

Validation performance is unexpectedly strong, while production performance deteriorates.

## Candidate objective

Identify and repair a point-in-time feature leakage issue without breaking the feature contract.

## Planned hidden evaluation

- unseen timestamps
- future-row perturbation
- shuffled records
- temporal boundary tests
- regression tests
- metric threshold

The reference implementation is not the only accepted solution.
