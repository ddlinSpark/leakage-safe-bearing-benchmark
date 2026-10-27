from typing import Optional

import torch
import torch.nn as nn


class AffineCalibrator(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_classes, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.scale + self.bias


class ConditionAwareCalibrator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_conditions: int,
        embedding_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_conditions, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_classes + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        final_linear = self.mlp[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def forward(self, logits: torch.Tensor, condition_ids: torch.Tensor) -> torch.Tensor:
        condition_emb = self.embedding(condition_ids)
        delta = self.mlp(torch.cat([logits, condition_emb], dim=1))
        return logits + delta
