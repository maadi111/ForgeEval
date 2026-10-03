"""Alternative hardened evaluation harness: Reference Solution B.

Features:
- Immutable tuple construction and isolated evaluation frames.
- Macro-F1 and Balanced Accuracy scoring for strict class calibration.
- Prediction entropy testing to detect constant or degenerate predictions.
- Dual-pass randomized permutation invariance verification.
- Jitter stability testing to isolate and reject memorized lookup tables.
- Cryptographic state digest integrity validation.
"""

import copy
import math

import numpy as np
from models import EvaluationReport, PredictorProtocol
from security import DatasetIntegrityVerifier
from sklearn.metrics import f1_score


class ModelEvaluator:
    """Alternative hardened model evaluation engine."""

    def __init__(self, dataset: dict):
        # Deep defensive copy on initialization
        self.dataset = copy.deepcopy(dataset)
        self.verifier = DatasetIntegrityVerifier(self.dataset)
        self.records = self.dataset.get("records", [])

    def evaluate(self, model: PredictorProtocol) -> EvaluationReport:
        """Run evaluation with calibration, invariance, and anti-tamper controls."""
        failures = []

        # Convert features to immutable tuples inside a fresh list
        X_clean = [list(r["features"]) for r in self.records]
        y_true = np.array([r["label"] for r in self.records], dtype=int)
        n = len(y_true)

        if n == 0:
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=["Empty dataset provided"],
            )

        # 1. Base inference
        try:
            preds_raw = model.predict(copy.deepcopy(X_clean))
            preds = np.array([int(p) for p in preds_raw], dtype=int)
        except Exception as e:
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=[f"Model execution error: {e}"],
            )

        if len(preds) != n:
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=[f"Output dimension mismatch: expected {n}, got {len(preds)}"],
            )

        # 2. Entropy and Diversity Check
        p1 = float(np.mean(preds == 1))
        p0 = 1.0 - p1
        if p1 <= 0.0 or p0 <= 0.0:
            entropy = 0.0
        else:
            entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))

        if entropy < 0.10:
            failures.append(
                f"Degenerate prediction distribution: entropy {entropy:.3f} indicates constant/trivial predictions"
            )

        # 3. Macro-F1 and Balanced Accuracy
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        pos_acc = float(np.mean(preds[y_true == 1] == 1)) if np.any(y_true == 1) else 0.0
        neg_acc = float(np.mean(preds[y_true == 0] == 0)) if np.any(y_true == 0) else 0.0
        balanced_acc = 0.5 * (pos_acc + neg_acc)

        if balanced_acc < 0.65 or macro_f1 < 0.55:
            failures.append(
                f"Evaluation failed calibration threshold: balanced_acc={balanced_acc:.3f}, macro_f1={macro_f1:.3f}"
            )

        # 4. Randomized Permutation Equivariance Check
        rng = np.random.default_rng(2026)
        perm = rng.permutation(n)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(n)

        X_perm = [copy.deepcopy(X_clean[i]) for i in perm]
        try:
            shuffled_preds_raw = model.predict(copy.deepcopy(X_perm))
            shuffled_preds = np.array([int(p) for p in shuffled_preds_raw], dtype=int)
            unshuffled = shuffled_preds[inv_perm]
            order_agreement = float(np.mean(unshuffled == preds))
        except Exception as e:
            order_agreement = 0.0
            failures.append(f"Permutation test failure: {e}")

        if order_agreement < 0.98:
            failures.append(f"Order exploit detected: output depends on batch order (agreement {order_agreement:.3f})")

        # 5. Continuous Jitter Perturbation (Catch Hash-Memorization)
        X_perturbed = []
        for row in X_clean:
            # Deterministic micro-jitter
            perturbed_row = [float(v) + (0.02 if idx % 2 == 1 else -0.02) for idx, v in enumerate(row)]
            X_perturbed.append(perturbed_row)

        try:
            jitter_preds_raw = model.predict(copy.deepcopy(X_perturbed))
            jitter_preds = np.array([int(p) for p in jitter_preds_raw], dtype=int)
            perturbation_agreement = float(np.mean(jitter_preds == preds))
        except Exception as e:
            perturbation_agreement = 0.0
            failures.append(f"Jitter test failure: {e}")

        if perturbation_agreement < 0.90:
            failures.append(
                f"Memorization shortcut detected: predictions collapse under feature jitter (agreement {perturbation_agreement:.3f})"
            )

        # 6. Integrity check
        tamper_ok = self.verifier.verify_integrity()
        if not tamper_ok:
            failures.append("Dataset tampering detected: evaluation fixture was modified in memory")

        # Scoring
        score_base = (balanced_acc * 40.0) + (macro_f1 * 20.0)
        score_inv = (order_agreement * 15.0) + (perturbation_agreement * 15.0)
        score_sec = 10.0 if tamper_ok else 0.0

        total_score = round(score_base + score_inv + score_sec, 2)
        if failures:
            total_score = min(total_score, 60.0)

        passed = (len(failures) == 0) and (total_score >= 80.0)

        return EvaluationReport(
            passed=passed,
            score=total_score,
            metrics={
                "balanced_accuracy": round(balanced_acc, 4),
                "macro_f1": round(macro_f1, 4),
                "entropy": round(entropy, 4),
                "order_agreement": round(order_agreement, 4),
                "perturbation_agreement": round(perturbation_agreement, 4),
            },
            failures=failures,
            tamper_detected=not tamper_ok,
            invariance_verified=(order_agreement >= 0.98 and perturbation_agreement >= 0.90),
            details={"n_samples": n},
        )
