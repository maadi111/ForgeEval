"""
Reference Solution A — Direct canonical ordering & natural batch processing.

Fix Strategy:
1. Enforces strict canonical feature ordering during preprocessing by looking up
   features by declared column name rather than dictionary iteration.
2. Eliminates unsafe sorting in predict_batch, processing vectors in original caller order.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any

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
        """FIX: Retrieve attributes strictly in self.features order."""
        return [float(payload[col]) for col in self.features]

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        vector = self.preprocess(payload)
        prob = float(self.model.predict_proba([vector])[0][1])
        return {
            "prediction": int(prob >= 0.5),
            "probability": round(prob, 4),
        }

    def predict_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """FIX: Process items in their natural, original caller order."""
        if not items:
            return []

        vectors = [self.preprocess(item) for item in items]
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
        return results
