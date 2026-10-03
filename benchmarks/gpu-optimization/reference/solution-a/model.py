"""PyTorch model definition for neural inference."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.head(h)
