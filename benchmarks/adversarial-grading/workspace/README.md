# Benchmark: Adversarial Evaluation & Anti-Cheat Grading Harness

## Scenario Description

Our production risk platform evaluates external ML models submitted by partner institutions before deploying them to live transaction scoring. A recent security audit uncovered significant vulnerabilities in `evaluator.py` and `security.py`:

1. **False Acceptance of Degenerate Models**: Candidate models with zero discriminative capability (such as constant predictors or uncalibrated majority-class estimators) currently receive high scores (>85%) and pass the deployment gate.
2. **State & Fixture Corruption**: Certain untrusted model submissions mutate input data structures in-place, causing unpredictable state leakage across test suites and false verification reports.
3. **Memorization & Overfitting Bypasses**: Submissions that memorize visible evaluation inputs or exploit fixed feature representations pass validation despite failing when exposed to slight input variations or real-world distribution nuances.
4. **Order Fragility**: Production traffic arrives asynchronously, but models that strictly rely on sequential input order are not detected during batch validation.

## Requirements

1. **Harden `evaluator.py`**:
   - Ensure the evaluation pipeline strictly isolates evaluation datasets and prevents any in-place mutation or tampering.
   - Replace naive metrics with balanced, calibrated metrics (e.g., balanced accuracy, ROC-AUC, or macro-F1) that prevent trivial constant-output gaming.
   - Verify behavioral invariance (such as permutation invariance and stability under bounded input perturbations).
   - Ensure clear, structured `EvaluationReport` outputs documenting passing status, composite score, metrics, and any detected exploit attempts.
2. **Harden `security.py`**:
   - Implement cryptographic integrity verification for evaluation fixtures to detect any tampering or modification before/after execution.
3. **Preserve Valid Generalization**:
   - Ensure genuine, well-trained machine learning models (e.g., logistic regression, tree ensembles, neural baselines) pass cleanly with high confidence and scores >= 80.0.

## Public Test Verification

Run the public test suite to verify your baseline contract:

```bash
pytest ../tests/public/ -v
```
