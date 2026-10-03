# ForgeEval — Complete Project Specification

## 1. Executive summary

ForgeEval is a benchmark and evaluation system for realistic ML engineering tasks. A task author defines a production-like failure, reproducible workspace, public tests, hidden tests, reference implementations, and a grading policy. A candidate solution is executed in isolation and evaluated by externally observable behavior rather than patch similarity.

## 2. Problem statement

Production ML engineering differs from simple deterministic coding tasks. A correct solution may use a different algorithm, architecture, preprocessing strategy, or optimization. Weak graders can be gamed through hardcoded outputs, reference memorization, test-specific branches, test modification, or assumptions about visible inputs.

ForgeEval addresses this with behavioral, hidden, and adversarial grading.

## 3. Functional requirements

### FR-01 Task registry
Store task ID, title, version, difficulty, category, repository template, dataset version, environment, public tests, hidden tests, grader, reference solutions, limits, and scoring policy.

### FR-02 Reproducible task packages
Each task contains:

```text
task.yaml
Dockerfile
workspace/
tests/public/
tests/hidden/
grader/
reference/
data/
```

### FR-03 Candidate execution
Execute untrusted candidate code in Docker with timeout, CPU/memory limits, optional GPU, disabled network by default, workspace mounts, and captured stdout/stderr/exit status.

### FR-04 Grading
Support correctness, regression, generalization, adversarial, performance, and integrity tests.

### FR-05 Multiple valid solutions
Do not require reproduction of the reference patch. Grade the behavioral contract.

### FR-06 Negative controls
Maintain known-bad implementations such as hardcoded outputs, constant predictions, test-name conditionals, input memorization, and test tampering. They must fail.

### FR-07 Score report
Example:

```json
{
  "task_id": "feature-leakage-v1",
  "submission_id": "example",
  "score": 87.5,
  "components": {
    "correctness": 40,
    "regression": 15,
    "generalization": 15,
    "adversarial": 12.5,
    "performance": 5
  },
  "failures": []
}
```

## 4. Default scoring

| Component | Weight |
|---|---:|
| Correctness | 40 |
| Regression safety | 15 |
| Generalization | 15 |
| Adversarial resistance | 15 |
| Performance | 10 |
| Integrity | 5 |

Weights are versioned and task-specific.

## 5. Benchmark specifications

### B1 — Point-in-Time Feature Leakage

Symptom: validation performance is unusually high while production performance collapses. Root cause: a feature uses information unavailable at prediction time. Candidate must repair the feature pipeline without breaking its contract.

Hidden evaluation: temporal boundaries, future-row perturbation, shuffled records, unseen timestamps, regression checks, and metric thresholds.

### B2 — Retrieval Drift

Symptom: retrieval quality deteriorates after distribution shift. Evaluate Recall@K, Precision@K, embedding statistics, unseen corpus behavior, and latency.

### B3 — Broken Model Serving

Symptom: production requests occasionally receive incorrect predictions. Potential causes include preprocessing mismatch, batch ordering, stale artifacts, fallback paths, or serialization. Evaluate batch invariance, request ordering, unseen requests, failure handling, correctness, and latency.

### B4 — Slow PyTorch Inference

Symptom: inference latency increases substantially. Valid solutions may use batching, inference_mode, mixed precision, optimized preprocessing, or reduced CPU/GPU transfers. Grade the outcome, not the implementation.

### B5 — Adversarial Grader

Evaluate whether the platform rejects hardcoded outputs, input-specific conditionals, test tampering, reference memorization, and output-order exploits while accepting the reference and independent valid solutions.

## 6. AI-agent evaluation

Optional flow:

```text
Task -> agent gets repository + instructions -> inspect -> modify -> test -> retry -> submit -> ForgeEval grade
```

Metrics include task success rate, score, runtime, iterations, test failures, adversarial failures, and optional token/cost metrics.

## 7. Versioning

Version tasks, datasets, graders, environments, and scoring policies independently. Example: `feature-leakage-v1`, `dataset-2026-10`, `grader-1.2`.

## 8. Security

Candidate containers should run non-root where practical, have no privileged mode, no network by default, CPU/memory limits, execution timeouts, isolated workspaces, and no access to hidden grader logic.

## 9. MVP completion criteria

API can create/list tasks; tasks execute in Docker; candidates can submit; public/hidden graders run; score reports are generated; reference and alternative solutions pass; shortcuts fail; results persist; CI passes; README demonstrates an end-to-end run.
