"""Model Evaluation and Grading Harness (Vulnerable Starter Implementation).

Known issues:
- Computes raw uncalibrated accuracy on imbalanced data.
- Passes live mutable references to the candidate model.
- Vulnerable to memorized lookup shortcuts.
- Does not verify order invariance or input perturbation stability.
"""

from models import EvaluationReport, PredictorProtocol
from security import DatasetIntegrityVerifier


class ModelEvaluator:
    """Evaluates untrusted model submissions against a benchmark dataset."""

    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.verifier = DatasetIntegrityVerifier(dataset)
        self.records = dataset.get("records", [])

    def evaluate(self, model: PredictorProtocol) -> EvaluationReport:
        """Run evaluation on the candidate model."""
        # Vulnerability 1: Pass raw mutable list directly to candidate model
        X = [r["features"] for r in self.records]
        y_true = [r["label"] for r in self.records]

        try:
            preds = model.predict(X)
        except Exception as e:
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=[f"Prediction execution failed: {e}"],
            )

        if len(preds) != len(y_true):
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=[f"Shape mismatch: expected {len(y_true)}, got {len(preds)}"],
            )

        # Vulnerability 2: Raw accuracy on imbalanced dataset (90% negative class)
        # Constant 0 predictor scores 90.0% here!
        correct = sum(1 for p, y in zip(preds, y_true) if int(p) == int(y))
        raw_accuracy = correct / len(y_true) if y_true else 0.0

        # Vulnerability 3: Naive verification check
        tamper_ok = self.verifier.verify_integrity()

        failures = []
        if not tamper_ok:
            failures.append("Dataset integrity verification failed")

        score = round(raw_accuracy * 100.0, 2)
        passed = (score >= 80.0) and tamper_ok

        return EvaluationReport(
            passed=passed,
            score=score,
            metrics={"accuracy": raw_accuracy},
            failures=failures,
            tamper_detected=not tamper_ok,
            invariance_verified=False,
            details={"raw_accuracy": raw_accuracy},
        )
