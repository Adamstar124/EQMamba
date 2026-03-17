# eqmamba/models/cnn1d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def match_length(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 x 的時間長度對齊到 ref（裁切或右側補零）"""
    tx = x.size(-1)
    tr = ref.size(-1)
    if tx == tr:
        return x
    if tx > tr:
        return x[..., :tr]
    return F.pad(x, (0, tr - tx))


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 7, norm: str = "gn", act: str = "silu", dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=1, padding=pad)

        if norm == "gn":
            g = 8 if out_ch >= 8 else 1
            self.norm = nn.GroupNorm(g, out_ch)
        elif norm == "bn":
            self.norm = nn.BatchNorm1d(out_ch)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown act: {act}")

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class DownBlock1D(nn.Module):
    """stride=2 下採樣：Conv(stride=2) + ConvBlock"""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 7,
        norm: str = "gn",
        act: str = "silu",
        dropout: float = 0.0,
        down_k: int | None = None,
        down_p: int | None = None,
    ):
        super().__init__()
        down_k = k if down_k is None else down_k
        down_p = (down_k // 2) if down_p is None else down_p
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=down_k, stride=2, padding=down_p)
        self.post = ConvBlock1D(out_ch, out_ch, k=k, norm=norm, act=act, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.post(x)
        return x


class UpBlock1D(nn.Module):
    """上採樣：Upsample×2 + ConvBlock（比 ConvTranspose 更穩）"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 7, norm: str = "gn", act: str = "silu", dropout: float = 0.0, mode: str = "nearest"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.conv = ConvBlock1D(in_ch, out_ch, k=k, norm=norm, act=act, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        return x


@dataclass(frozen=True)
class StridedCNN2xConfig:
    in_ch: int = 3
    k: int = 7
    down_k: int | None = None
    down_p: int | None = None
    skip_refine_k1: bool = False
    norm: str = "gn"
    act: str = "silu"
    dropout: float = 0.0
    # 兩次下採樣：C0 -> C1 -> C2
    chs: Tuple[int, int, int] = (16, 32, 64)


@dataclass(frozen=True)
class UNetSkips2x:
    """
    標準 U-Net skips（2層）
    skip0: (B,C0,6000)
    skip1: (B,C1,3000)
    """
    skip0: torch.Tensor
    skip1: torch.Tensor


class StridedUNetEncoder2x(nn.Module):
    """
    2x stride downsample encoder

    input:  (B,3,6000)
    output: z (B,C2,1500), skips(skip0, skip1)
    """
    def __init__(self, cfg: StridedCNN2xConfig):
        super().__init__()
        c0, c1, c2 = cfg.chs
        self.stem  = ConvBlock1D(cfg.in_ch, c0, k=cfg.k, norm=cfg.norm, act=cfg.act, dropout=cfg.dropout)   # 6000
        self.down1 = DownBlock1D(
            c0, c1, k=cfg.k, norm=cfg.norm, act=cfg.act, dropout=cfg.dropout,
            down_k=cfg.down_k, down_p=cfg.down_p,
        )                                                                                                    # 3000
        self.down2 = DownBlock1D(
            c1, c2, k=cfg.k, norm=cfg.norm, act=cfg.act, dropout=cfg.dropout,
            down_k=cfg.down_k, down_p=cfg.down_p,
        )                                                                                                    # 1500

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, UNetSkips2x]:
        skip0 = self.stem(x)        # (B,C0,6000)
        skip1 = self.down1(skip0)   # (B,C1,3000)
        z     = self.down2(skip1)   # (B,C2,1500)
        return z, UNetSkips2x(skip0=skip0, skip1=skip1)


class StridedUNetDecoder2x(nn.Module):
    """
    2x upsample decoder with standard U-Net skip fusion

    input:  z (B,C2,1500), skips(skip0, skip1)
    output: feat (B,C0,6000)
    """
    def __init__(self, cfg: StridedCNN2xConfig):
        super().__init__()
        c0, c1, c2 = cfg.chs
        if cfg.skip_refine_k1:
            self.skip1_refine = ConvBlock1D(c1, c1, k=1, norm="none", act=cfg.act, dropout=0.0)
            self.skip0_refine = ConvBlock1D(c0, c0, k=1, norm="none", act=cfg.act, dropout=0.0)
        else:
            self.skip1_refine = nn.Identity()
            self.skip0_refine = nn.Identity()

        # 1500 -> 3000
        self.up1 = UpBlock1D(c2, c1, k=cfg.k, norm=cfg.norm, act=cfg.act, dropout=cfg.dropout)
        self.fuse1 = ConvBlock1D(c1 + c1, c1, k=cfg.k, norm=cfg.norm, act=cfg.act, dropout=cfg.dropout)

        # 3000 -> 6000
        self.up2 = UpBlock1D(c1, c0, k=cfg.k, norm=cfg.norm, act=cfg.act, dropout=cfg.dropout)
        self.fuse2 = ConvBlock1D(c0 + c0, c0, k=cfg.k, norm=cfg.norm, act=cfg.act, dropout=cfg.dropout)

    def forward(self, z: torch.Tensor, skips: UNetSkips2x) -> torch.Tensor:
        skip1 = self.skip1_refine(skips.skip1)
        skip0 = self.skip0_refine(skips.skip0)

        # stage 1: align to skip1
        x = self.up1(z)
        x = match_length(x, skip1)
        x = torch.cat([x, skip1], dim=1)
        x = self.fuse1(x)

        # stage 2: align to skip0
        x = self.up2(x)
        x = match_length(x, skip0)
        x = torch.cat([x, skip0], dim=1)
        x = self.fuse2(x)

        return x
