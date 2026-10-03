"""Harden evaluation harness: Reference Solution A.

Features:
- Defensive isolation and deepcopy of all test records.
- Cryptographic SHA-256 fixture integrity validation.
- Balanced accuracy and ROC-AUC evaluation (immune to class imbalance).
- Permutation invariance testing (detects order exploits).
- Epsilon jitter perturbation stability (detects hash memorization shortcuts).
- Prediction entropy / diversity checks (detects constant predictors).
"""

import copy
import random

import numpy as np
from models import EvaluationReport, PredictorProtocol
from security import DatasetIntegrityVerifier
from sklearn.metrics import roc_auc_score


class ModelEvaluator:
    """Hardened behavioral and adversarial model evaluation harness."""

    def __init__(self, dataset: dict):
        self.dataset = copy.deepcopy(dataset)
        self.verifier = DatasetIntegrityVerifier(self.dataset)
        self.records = self.dataset.get("records", [])

    def evaluate(self, model: PredictorProtocol) -> EvaluationReport:
        """Run complete behavioral evaluation suite against untrusted model."""
        failures = []

        # 1. Defensive deepcopy of evaluation inputs
        X_clean = copy.deepcopy([r["features"] for r in self.records])
        y_true = np.array([r["label"] for r in self.records], dtype=int)
        n_samples = len(y_true)

        if n_samples == 0:
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=["Empty evaluation dataset"],
            )

        # 2. Base inference execution
        try:
            preds_raw = model.predict(copy.deepcopy(X_clean))
            preds = np.array([int(p) for p in preds_raw], dtype=int)
        except Exception as e:
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=[f"Base prediction execution failed: {e}"],
            )

        if len(preds) != n_samples:
            return EvaluationReport(
                passed=False,
                score=0.0,
                failures=[f"Output dimension mismatch: expected {n_samples}, got {len(preds)}"],
            )

        # 3. Probability estimation execution
        try:
            probs_raw = model.predict_proba(copy.deepcopy(X_clean))
            probs = np.array([float(p) for p in probs_raw], dtype=float)
        except Exception:
            probs = preds.astype(float)

        # 4. Check for constant / degenerate predictions
        pred_variance = float(np.var(preds))
        if pred_variance < 1e-6:
            failures.append("Constant prediction shortcut detected: zero output variance")

        # 5. Balanced Accuracy & Metric Computation
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        tpr = np.mean(preds[pos_mask] == 1) if np.any(pos_mask) else 0.0
        tnr = np.mean(preds[neg_mask] == 0) if np.any(neg_mask) else 0.0
        balanced_acc = float(0.5 * (tpr + tnr))

        try:
            auc = float(roc_auc_score(y_true, probs))
        except Exception:
            auc = balanced_acc

        if balanced_acc < 0.65:
            failures.append(f"Insufficient discriminative power: balanced accuracy {balanced_acc:.3f} < 0.65")

        # 6. Permutation Equivariance / Order Invariance Test
        rng = random.Random(1337)
        perm = list(range(n_samples))
        rng.shuffle(perm)
        inv_perm = [0] * n_samples
        for idx, p_idx in enumerate(perm):
            inv_perm[p_idx] = idx

        X_shuffled = [copy.deepcopy(X_clean[i]) for i in perm]
        try:
            shuffled_preds_raw = model.predict(copy.deepcopy(X_shuffled))
            shuffled_preds = np.array([int(p) for p in shuffled_preds_raw], dtype=int)
            restored_preds = shuffled_preds[inv_perm]
            order_agreement = float(np.mean(restored_preds == preds))
        except Exception as e:
            order_agreement = 0.0
            failures.append(f"Order permutation test execution failed: {e}")

        if order_agreement < 0.98:
            failures.append(
                f"Order exploit detected: model output depends on batch input sequence (agreement {order_agreement:.3f})"
            )

        # 7. Perturbation Stability / Anti-Memorization Test
        # Add bounded epsilon noise (0.02) to numeric features
        X_jittered = []
        for row in X_clean:
            jittered_row = [float(val) + (0.02 if (i % 2 == 0) else -0.02) for i, val in enumerate(row)]
            X_jittered.append(jittered_row)

        try:
            jitter_preds_raw = model.predict(copy.deepcopy(X_jittered))
            jitter_preds = np.array([int(p) for p in jitter_preds_raw], dtype=int)
            perturbation_agreement = float(np.mean(jitter_preds == preds))
        except Exception as e:
            perturbation_agreement = 0.0
            failures.append(f"Perturbation test execution failed: {e}")

        if perturbation_agreement < 0.90:
            failures.append(
                f"Memorization shortcut detected: model output collapses under slight input perturbation (agreement {perturbation_agreement:.3f})"
            )

        # 8. Cryptographic Tampering Verification
        tamper_ok = self.verifier.verify_integrity()
        if not tamper_ok:
            failures.append("Dataset tampering detected: evaluation fixture was mutated in-place")

        # Composite Scoring Policy
        # Correctness: balanced accuracy (40 pts)
        correctness_pts = min(40.0, max(0.0, (balanced_acc - 0.5) / 0.4 * 40.0))
        # Discrimination: AUC (20 pts)
        auc_pts = min(20.0, max(0.0, (auc - 0.5) / 0.4 * 20.0))
        # Invariance: Order (15 pts) + Perturbation (15 pts)
        invariance_pts = (order_agreement * 15.0) + (perturbation_agreement * 15.0)
        # Integrity: (10 pts)
        integrity_pts = 10.0 if tamper_ok else 0.0

        total_score = round(correctness_pts + auc_pts + invariance_pts + integrity_pts, 2)
        if failures:
            total_score = min(total_score, 60.0)

        passed = (len(failures) == 0) and (total_score >= 80.0)

        return EvaluationReport(
            passed=passed,
            score=total_score,
            metrics={
                "balanced_accuracy": round(balanced_acc, 4),
                "roc_auc": round(auc, 4),
                "order_agreement": round(order_agreement, 4),
                "perturbation_agreement": round(perturbation_agreement, 4),
            },
            failures=failures,
            tamper_detected=not tamper_ok,
            invariance_verified=(order_agreement >= 0.98 and perturbation_agreement >= 0.90),
            details={
                "n_samples": n_samples,
                "tpr": round(float(tpr), 4),
                "tnr": round(float(tnr), 4),
            },
        )
