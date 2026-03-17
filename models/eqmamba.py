# eqmamba/models/eqmamba.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from .cnn1d import StridedCNN2xConfig, StridedUNetEncoder2x
from .blocks import MambaBlockBTC, PerformerBlockBTC, RMSNorm
from .heads import EQMambaHead2x


@dataclass(frozen=True)
class EQMamba2xConfig:
    in_samples: int = 6000
    d_model: int = 64          # 建議等於 cnn bottleneck C2
    d_state: int = 16
    core_mamba_nums: int = 4
    decode_mamba_nums: int = 2
    performer_heads: int = 4
    drop_rate: float = 0.0
    norm: str = "layernorm"    # or "rmsnorm"

    # ✅ 固定輸出/label 順序：[P, S, Event]
    head_names: List[str] = None

    cnn: StridedCNN2xConfig = StridedCNN2xConfig(
        in_ch=3,
        k=7,
        norm="gn",
        act="silu",
        dropout=0.0,
        chs=(16, 32, 64),   # C2=64
    )

    def __post_init__(self):
        if self.head_names is None:
            object.__setattr__(self, "head_names", ["P", "S", "event"])


class EQMamba2x(nn.Module):
    """
    x: (B,3,6000)
    logits3 out: (B,3,6000)  channel=[P,S,Event]
    """
    def __init__(self, cfg: EQMamba2xConfig):
        super().__init__()
        self.cfg = cfg
        self.head_names = list(cfg.head_names)  # 保存順序

        # --- CNN Encoder (BCT) ---
        self.encoder = StridedUNetEncoder2x(cfg.cnn)

        # --- Core in BTC ---
        self.performer_start = PerformerBlockBTC(
            dim=cfg.d_model, depth=1, heads=cfg.performer_heads,
            dim_head=cfg.d_model // cfg.performer_heads, causal=False
        )
        self.performer_end = PerformerBlockBTC(
            dim=cfg.d_model, depth=1, heads=cfg.performer_heads,
            dim_head=cfg.d_model // cfg.performer_heads, causal=False
        )

        def make_norm():
            if cfg.norm == "layernorm":
                return nn.LayerNorm(cfg.d_model)
            elif cfg.norm == "rmsnorm":
                return RMSNorm(cfg.d_model)
            else:
                raise ValueError(cfg.norm)

        self.core_blocks = nn.ModuleList([
            nn.ModuleDict({
                "mamba": MambaBlockBTC(d_model=cfg.d_model, d_state=cfg.d_state, drop_rate=cfg.drop_rate),
                "norm": make_norm(),
            })
            for _ in range(cfg.core_mamba_nums)
        ])

        # --- Heads（✅ 用 ModuleList 固定順序，Head 不需要 name） ---
        self.heads = nn.ModuleList([
            EQMambaHead2x(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                decode_mamba_nums=cfg.decode_mamba_nums,
                performer_heads=cfg.performer_heads,
                cnn_cfg=cfg.cnn,
                norm=cfg.norm,
                drop_rate=cfg.drop_rate,
            )
            for _ in self.head_names
        ])

        # safety check: d_model must match cnn bottleneck ch
        c2 = cfg.cnn.chs[2]
        if cfg.d_model != c2:
            raise ValueError(
                f"d_model({cfg.d_model}) must equal cnn bottleneck channels C2({c2}). "
                f"Either set d_model={c2} or add a projection layer."
            )

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        # CNN encode: (B,3,6000) -> z(B,C2,1500), skips
        z_bct, skips = self.encoder(x_bct)

        # to BTC for core: (B,C2,1500) -> (B,1500,C2)
        en_btc = z_bct.transpose(1, 2).contiguous()

        h = self.performer_start(en_btc)
        for blk in self.core_blocks:
            y = blk["mamba"](h)
            y = blk["norm"](y)
            h = h + y
        h = self.performer_end(h)

        # ✅ heads 輸出依序對應 self.head_names（預設 [P,S,event]）
        logits_list = [head(h, en_btc, skips) for head in self.heads]  # each (B,6000)
        logits3 = torch.stack(logits_list, dim=1)  # (B,3,6000)
        return logits3
