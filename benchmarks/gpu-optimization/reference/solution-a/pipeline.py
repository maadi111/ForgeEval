"""Optimized Neural Inference Pipeline: Reference Solution A.

Optimizations:
- Vectorized batching: converts full request batch into a single contiguous 2D float32 tensor.
- Evaluates within torch.inference_mode() to eliminate autograd graph tracking.
- model.eval() sets deterministic LayerNorm / Dropout behavior.
- Vectorized softmax and argmax on GPU/CPU tensor before single array extraction.
"""

import pathlib
from typing import Any

import numpy as np
import torch
from model import NeuralClassifier


class InferencePipeline:
    """Optimized inference pipeline utilizing vectorized batching and inference_mode."""

    def __init__(self, weights_path: str | pathlib.Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralClassifier(in_features=64, hidden_dim=128, out_features=2)

        if weights_path is not None:
            weights_file = pathlib.Path(weights_path)
            if weights_file.exists():
                state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict_one(self, features: list[float]) -> dict[str, Any]:
        """Process single prediction using inference mode."""
        with torch.inference_mode():
            t_in = torch.tensor([features], dtype=torch.float32, device=self.device)
            logits = self.model(t_in)
            probs = torch.softmax(logits, dim=-1)[0]
            p_val = float(probs[1].item())
            pred = 1 if p_val >= 0.5 else 0
            return {"probability": round(p_val, 5), "prediction": pred}

    def predict_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Vectorized batch prediction."""
        if not batch:
            return []

        # Extract features into bulk numpy array
        features_arr = np.array([req["features"] for req in batch], dtype=np.float32)

        with torch.inference_mode():
            t_in = torch.from_numpy(features_arr).to(self.device)
            logits = self.model(t_in)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        results = []
        for idx, req in enumerate(batch):
            p = float(probs[idx])
            results.append({
                "request_id": req.get("request_id", ""),
                "probability": round(p, 5),
                "prediction": 1 if p >= 0.5 else 0,
            })
        return results
