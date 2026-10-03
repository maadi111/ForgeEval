"""Generate PyTorch neural model weights and synthetic benchmark requests for B4."""

import json
import pathlib
import sys

import numpy as np


def generate_benchmark_assets(data_dir: pathlib.Path) -> None:
    """Generate model weights and test requests."""
    import torch
    from torch import nn

    data_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    class ResidualBlock(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            res = x
            x = self.act(self.ln1(self.fc1(x)))
            x = self.ln2(self.fc2(x))
            return self.act(x + res)

    class NeuralClassifier(nn.Module):
        def __init__(self, in_features: int = 64, hidden_dim: int = 128, out_features: int = 2):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.block1 = ResidualBlock(hidden_dim)
            self.block2 = ResidualBlock(hidden_dim)
            self.block3 = ResidualBlock(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_features)

        def forward(self, x):
            h = self.proj(x)
            h = self.block1(h)
            h = self.block2(h)
            h = self.block3(h)
            return self.head(h)

    model = NeuralClassifier(in_features=64, hidden_dim=128, out_features=2)
    model.eval()

    # Save weights
    weights_path = data_dir / "model_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved neural model weights -> {weights_path}")

    # Generate 256 test requests
    n_requests = 256
    requests_data = []
    features_matrix = np.random.randn(n_requests, 64).astype(np.float32)

    for i in range(n_requests):
        requests_data.append({
            "request_id": f"req_{i:04d}",
            "features": [round(float(v), 5) for v in features_matrix[i]],
        })

    requests_path = data_dir / "eval_requests.json"
    with open(requests_path, "w", encoding="utf-8") as f:
        json.dump({"n_requests": n_requests, "requests": requests_data}, f, indent=2)
    print(f"Saved {n_requests} test requests -> {requests_path}")

    # Generate ground truth predictions
    with torch.no_grad():
        t_in = torch.from_numpy(features_matrix)
        logits = model(t_in)
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy().tolist()
        preds = (np.array(probs) >= 0.5).astype(int).tolist()

    gt_data = {
        "probabilities": [round(float(p), 5) for p in probs],
        "predictions": preds,
    }
    gt_path = data_dir / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)
    print(f"Saved ground truth predictions -> {gt_path}")


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parent
    if len(sys.argv) > 2 and sys.argv[1] == "--output_dir":
        out = pathlib.Path(sys.argv[2])
    generate_benchmark_assets(out)
