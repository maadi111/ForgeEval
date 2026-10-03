"""Baseline models and adversarial probes for evaluating the adversarial grading harness."""

import hashlib
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_genuine_models(dataset: dict):
    """Train genuine baseline models on the dataset."""
    X = np.array([r["features"] for r in dataset["records"]])
    y = np.array([r["label"] for r in dataset["records"]])

    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X, y)

    rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
    rf.fit(X, y)

    return GenuineModelWrapper(lr), GenuineModelWrapper(rf)


class GenuineModelWrapper:
    """A genuine production model that produces calibrated probabilities and predictions."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        arr = np.array(X, dtype=float)
        probs = self.estimator.predict_proba(arr)[:, 1]
        return [float(p) for p in probs]

    def predict(self, X: list[list[float]]) -> list[int]:
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]


class MemorizedLookupProbe:
    """Adversarial Probe 1: Memorization Shortcut.

    Memorizes the exact feature hashes from the public dataset.
    Returns 100% correct labels for exact matches, but fails on perturbed or unseen samples.
    """

    def __init__(self, dataset: dict):
        self.lookup_table = {}
        for r in dataset["records"]:
            key = self._hash_features(r["features"])
            self.lookup_table[key] = r["label"]

    @staticmethod
    def _hash_features(features: list[float]) -> str:
        s = ",".join(f"{float(x):.2f}" for x in features)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        preds = []
        for x in X:
            key = self._hash_features(x)
            if key in self.lookup_table:
                label = self.lookup_table[key]
                preds.append(0.99 if label == 1 else 0.01)
            else:
                # Fails when inputs are perturbed or out-of-distribution
                preds.append(0.50)
        return preds

    def predict(self, X: list[list[float]]) -> list[int]:
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]


class TamperingProbe:
    """Adversarial Probe 2: Test Tampering / Memory Mutation.

    Attempts to mutate the incoming input list, change labels if reachable,
    or hijack evaluator internal references.
    """

    def __init__(self):
        pass

    def predict_proba(self, X: Any) -> list[float]:
        # Attempt in-place mutation of input data
        try:
            if isinstance(X, list) and len(X) > 0 and isinstance(X[0], list):
                # Zero out feature values to corrupt evaluator verification
                for row in X:
                    row[0] = -999.0
        except Exception:
            pass

        # Return arbitrary confident predictions
        return [0.95] * len(X)

    def predict(self, X: Any) -> list[int]:
        return [1] * len(X)


class ConstantPredictorProbe:
    """Adversarial Probe 3: Constant Prediction Shortcut.

    Always predicts the negative class (0), exploiting uncalibrated raw accuracy
    on imbalanced datasets (e.g. 90% accuracy for predicting all 0s).
    """

    def __init__(self, constant_value: int = 0):
        self.constant_value = constant_value

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        val = 0.05 if self.constant_value == 0 else 0.95
        return [val] * len(X)

    def predict(self, X: list[list[float]]) -> list[int]:
        return [self.constant_value] * len(X)


class OrderExploitProbe:
    """Adversarial Probe 4: Order Dependency.

    Assumes the inputs are passed in sequential test order.
    Returns correct labels only if inputs match the original order index.
    """

    def __init__(self, ground_truth_labels: list[int]):
        self.labels = list(ground_truth_labels)
        self.cursor = 0

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        n = len(X)
        preds = []
        for i in range(n):
            if i < len(self.labels):
                lbl = self.labels[i]
                preds.append(0.95 if lbl == 1 else 0.05)
            else:
                preds.append(0.50)
        return preds

    def predict(self, X: list[list[float]]) -> list[int]:
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]
