"""
Stage 1: the contrastive alignment projection head f_theta.

Maps OPERA-CT's 768-dim audio embedding into the LLM's 2560-dim (Phi-2)
semantic space, so audio and text vectors become directly comparable.

Paper spec: Linear -> LayerNorm -> ReLU -> Dropout(0.1) -> Linear
  768 -> 1024 -> 2560
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def from_config(cfg):
        return ProjectionHead(
            in_dim=cfg.audio_dim,
            hidden_dim=cfg.proj_hidden_dim,
            out_dim=cfg.llm_hidden_dim,
            dropout=cfg.proj_dropout,
        )
