"""
Model Serving Pipeline — credit risk evaluation.

Production Symptom:
Single unit tests pass, but customer complaints report incorrect scoring under
concurrent or batched traffic.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any

# Canonical feature names expected by the trained model
FEATURE_NAMES: list[str] = [
    "age",
    "income",
    "credit_score",
    "debt_to_income",
    "num_accounts",
    "recent_inquiries",
    "employment_years",
    "loan_amount",
]


class ModelServer:
    def __init__(self, model_path: str | pathlib.Path | None = None) -> None:
        if model_path is None:
            model_path = pathlib.Path(__file__).parent / "model.pkl"
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.features = artifact.get("features", FEATURE_NAMES)

    def preprocess(self, payload: dict[str, Any]) -> list[float]:
        """Convert request JSON dictionary into a numerical feature vector.

        BUG 1 (intentional): Iterates over payload.keys() directly rather than
        enforcing self.features order. If client payload keys are ordered differently,
        features become silently misaligned!
        """
        return [float(payload[k]) for k in payload if k in self.features]

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score a single incoming request."""
        vector = self.preprocess(payload)
        prob = float(self.model.predict_proba([vector])[0][1])
        prediction = int(prob >= 0.5)
        return {
            "prediction": prediction,
            "probability": round(prob, 4),
        }

    def predict_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a batch of incoming requests.

        BUG 2 (intentional): Batched requests are sorted by loan_amount for
        cache/quantization optimization, but predictions are returned directly
        in sorted order rather than the caller's original request order!
        """
        if not items:
            return []

        # Sort items by loan_amount to group similar profiles
        sorted_items = sorted(items, key=lambda x: x.get("loan_amount", 0))

        # Preprocess each item
        vectors = [self.preprocess(item) for item in sorted_items]

        # Model inference
        probs = self.model.predict_proba(vectors)[:, 1]

        results = []
        for p in probs:
            p_val = float(p)
            results.append(
                {
                    "prediction": int(p_val >= 0.5),
                    "probability": round(p_val, 4),
                }
            )

        # Returning results in sorted order instead of caller's original order!
        return results
