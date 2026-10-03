"""Alternative Optimized Neural Inference Pipeline: Reference Solution B.

Optimizations:
- TorchScript JIT Tracing: Compiles model forward graph for optimized runtime execution.
- Dynamic Chunked Vector Batching with pre-allocated tensors.
- Execution under torch.inference_mode().
"""

import pathlib
from typing import Any

import numpy as np
import torch
from model import NeuralClassifier


class InferencePipeline:
    """Alternative optimized inference pipeline utilizing TorchScript JIT compilation."""

    def __init__(self, weights_path: str | pathlib.Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = NeuralClassifier(in_features=64, hidden_dim=128, out_features=2)

        if weights_path is not None:
            weights_file = pathlib.Path(weights_path)
            if weights_file.exists():
                state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
                base_model.load_state_dict(state_dict)

        base_model.to(self.device)
        base_model.eval()

        # JIT Trace with dummy input
        dummy_input = torch.zeros((1, 64), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            self.model = torch.jit.trace(base_model, dummy_input)

    def predict_one(self, features: list[float]) -> dict[str, Any]:
        """Process single prediction with JIT model."""
        with torch.inference_mode():
            t_in = torch.tensor([features], dtype=torch.float32, device=self.device)
            logits = self.model(t_in)
            probs = torch.softmax(logits, dim=-1)[0]
            p_val = float(probs[1].item())
            pred = 1 if p_val >= 0.5 else 0
            return {"probability": round(p_val, 5), "prediction": pred}

    def predict_batch(self, batch: list[dict[str, Any]], chunk_size: int = 128) -> list[dict[str, Any]]:
        """Chunked vectorized batch prediction with TorchScript."""
        if not batch:
            return []

        all_probs = []
        n_total = len(batch)

        with torch.inference_mode():
            for i in range(0, n_total, chunk_size):
                chunk = batch[i : i + chunk_size]
                feat_chunk = np.array([r["features"] for r in chunk], dtype=np.float32)
                t_in = torch.from_numpy(feat_chunk).to(self.device)
                logits = self.model(t_in)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())

        results = []
        for idx, req in enumerate(batch):
            p = float(all_probs[idx])
            results.append({
                "request_id": req.get("request_id", ""),
                "probability": round(p, 5),
                "prediction": 1 if p >= 0.5 else 0,
            })
        return results
