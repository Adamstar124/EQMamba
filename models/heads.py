# eqmamba/models/heads.py
from __future__ import annotations

import torch
from torch import nn

from .cnn1d import StridedCNN2xConfig, StridedUNetDecoder2x, UNetSkips2x
from .blocks import MambaBlockBTC, PerformerBlockBTC, RMSNorm


class EQMambaHead2x(nn.Module):
    """
    不再需要 name（因為主模型會用 ModuleList 固定順序 [P,S,Event]）
    主幹：BTC
    head decode：BTC
    回到原長度：BCT（用 StridedUNetDecoder2x）
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        decode_mamba_nums: int,
        performer_heads: int,
        cnn_cfg: StridedCNN2xConfig,
        norm: str = "layernorm",  # or "rmsnorm"
        drop_rate: float = 0.0,
    ):
        super().__init__()

        def make_norm():
            if norm == "layernorm":
                return nn.LayerNorm(d_model)
            elif norm == "rmsnorm":
                return RMSNorm(d_model)
            else:
                raise ValueError(norm)

        self.decode_blocks = nn.ModuleList([
            nn.ModuleDict({
                "mamba": MambaBlockBTC(d_model=d_model, d_state=d_state, drop_rate=drop_rate),
                "norm": make_norm(),
            })
            for _ in range(decode_mamba_nums)
        ])

        self.decode_performer = PerformerBlockBTC(
            dim=d_model,
            depth=1,
            heads=performer_heads,
            dim_head=d_model // performer_heads,
            causal=False,
        )

        # head 自己的 decoder（BCT）
        self.decoder = StridedUNetDecoder2x(cnn_cfg)

        # 最後產生 logits
        self.out_conv = nn.Conv1d(
            in_channels=cnn_cfg.chs[0],
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, h_btc: torch.Tensor, en_btc: torch.Tensor, skips: UNetSkips2x) -> torch.Tensor:
        """
        h_btc, en_btc: (B,T,C=d_model)
        skips: UNetSkips2x (skip0/skip1 in BCT)
        return logits: (B,T_out=6000)
        """
        x = h_btc
        for blk in self.decode_blocks:
            y = blk["mamba"](x)
            y = blk["norm"](y)
            x = x + y

        x = self.decode_performer(x)

        # BTC skip 融合
        x = x + en_btc

        # BTC -> BCT bottleneck feature
        x_bct = x.transpose(1, 2).contiguous()   # (B,C2,1500) 若 d_model=C2

        feat = self.decoder(x_bct, skips)         # (B,C0,6000)
        logits = self.out_conv(feat).squeeze(1)   # (B,6000)
        return logits
