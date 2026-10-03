"""Neural Inference Pipeline (Slow / Unoptimized Starter Implementation).

Symptoms:
- Unbatched loop: executes single-sample forward pass in Python for-loop.
- Leaves autograd active, generating unnecessary computation graphs and memory allocations.
- Redundant tensor allocations per sample.
- Breaches production latency SLA.
"""

import pathlib
from typing import Any

import torch
from model import NeuralClassifier


class InferencePipeline:
    """Manages neural risk model inference."""

    def __init__(self, weights_path: str | pathlib.Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralClassifier(in_features=64, hidden_dim=128, out_features=2)

        if weights_path is not None:
            weights_file = pathlib.Path(weights_path)
            if weights_file.exists():
                state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        # Performance bug: forgot model.eval() and leaves model in training mode

    def predict_one(self, features: list[float]) -> dict[str, Any]:
        """Process a single request."""
        # Performance bug: un-cached tensor creation and autograd tracking
        t_in = torch.tensor([features], dtype=torch.float32, device=self.device)
        logits = self.model(t_in)
        probs = torch.softmax(logits, dim=-1)[0]
        p_val = float(probs[1].item())
        pred = 1 if p_val >= 0.5 else 0
        return {"probability": p_val, "prediction": pred}

    def predict_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a batch of requests.

        Performance bug: processes items in an unbatched Python loop, incurring
        repeated Python-C++ boundary crossings, redundant kernel launches, and autograd overhead.
        """
        results = []
        for req in batch:
            req_id = req.get("request_id", "")
            features = req.get("features", [])
            pred_dict = self.predict_one(features)
            results.append({
                "request_id": req_id,
                "probability": round(pred_dict["probability"], 5),
                "prediction": pred_dict["prediction"],
            })
        return results
