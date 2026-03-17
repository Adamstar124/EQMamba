# eqmamba/models/blocks.py
from __future__ import annotations

import torch
from torch import nn

from mamba_ssm import Mamba2
from performer_pytorch import Performer

class MambaBlockBTC(nn.Module):
    """
    x: (B, T, C) -> (B, T, C)
    """
    def __init__(self, d_model: int, d_state: int, drop_rate: float = 0.0):
        super().__init__()
        self.mamba = Mamba2(d_model, d_state)
        self.dropout = nn.Dropout(drop_rate) if drop_rate and drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.mamba(x)
        x = self.dropout(x)
        return x


class PerformerBlockBTC(nn.Module):
    """
    x: (B, T, C) -> (B, T, C)
    """
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, causal: bool):
        super().__init__()
        self.performer = Performer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.performer(x)


class RMSNorm(nn.Module):
    """
    可選：如果你想換成 RMSNorm（常見於序列模型）
    不想換就用 nn.LayerNorm(C) 也完全 OK。
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale
