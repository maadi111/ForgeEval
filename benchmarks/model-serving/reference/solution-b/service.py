"""
Reference Solution B — Inverse permutation mapping & schema-safe extraction.

Fix Strategy:
1. Retains bucketed batch sorting optimization, but tracks original indices and
   inverts the permutation before returning, ensuring 100% batch ordering invariance.
2. Uses schema-safe dictionary extraction with defaults.
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
        """FIX: Extract features strictly via canonical keys."""
        return [float(payload.get(k, 0.0)) for k in self.features]

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        vector = self.preprocess(payload)
        prob = float(self.model.predict_proba([vector])[0][1])
        return {
            "prediction": int(prob >= 0.5),
            "probability": round(prob, 4),
        }

    def predict_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """FIX: Maintain sorted grouping but invert permutation before returning."""
        if not items:
            return []

        # Tag each item with its original index
        indexed_items = list(enumerate(items))
        # Sort for batch execution efficiency
        indexed_items.sort(key=lambda pair: pair[1].get("loan_amount", 0))

        original_indices = [pair[0] for pair in indexed_items]
        vectors = [self.preprocess(pair[1]) for pair in indexed_items]

        probs = self.model.predict_proba(vectors)[:, 1]

        # Invert permutation back to caller order
        results: list[dict[str, Any] | None] = [None] * len(items)
        for orig_idx, p in zip(original_indices, probs):
            p_val = float(p)
            results[orig_idx] = {
                "prediction": int(p_val >= 0.5),
                "probability": round(p_val, 4),
            }

        return [r for r in results if r is not None]
