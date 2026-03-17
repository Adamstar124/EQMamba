# eqmamba/training/losses.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Any, Optional, Sequence

import torch
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class LossWeights3:
    """
    channel-wise loss weights for (B,3,T) with channel order [P, S, Event]
    """
    P: float = 0.3
    S: float = 0.5
    event: float = 0.2

    def as_tensor(self, device=None, dtype=None) -> torch.Tensor:
        t = torch.tensor([self.P, self.S, self.event], device=device, dtype=dtype)
        return t


@dataclass(frozen=True)
class Y3Order:
    """
    定義 batch["y"] 的 channel 順序（你目前是 [P,S,Event]）
    """
    P: int = 0
    S: int = 1
    event: int = 2


# ----------------------------
# Helpers
# ----------------------------
def _check_logits_y3(logits3: torch.Tensor) -> None:
    if not torch.is_tensor(logits3):
        raise TypeError(f"logits3 must be torch.Tensor, got {type(logits3)}")
    if logits3.ndim != 3 or logits3.size(1) != 3:
        raise ValueError(f"Expected logits3 shape (B,3,T), got {tuple(logits3.shape)}")


def _get_targets_y3(batch: Mapping[str, Any], y_key: str = "y") -> torch.Tensor:
    if y_key not in batch:
        raise KeyError(f"Batch missing '{y_key}'. Keys={list(batch.keys())}")
    y = batch[y_key]
    if not torch.is_tensor(y):
        raise TypeError(f"batch['{y_key}'] must be torch.Tensor, got {type(y)}")
    if y.ndim != 3 or y.size(1) != 3:
        raise ValueError(f"Expected batch['{y_key}'] shape (B,3,T), got {tuple(y.shape)}")
    return y.float()


# ----------------------------
# Main losses (tensor-only)
# ----------------------------
def bce_logits_loss_y3_tensor(
    logits3: torch.Tensor,
    batch: Mapping[str, Any],
    weights: LossWeights3 = LossWeights3(),
    y_key: str = "y",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    logits3: (B,3,T) channel=[P,S,Event]
    batch['y']: (B,3,T) same order
    用 BCEWithLogits，然後對三個 channel 做加權和
    """
    _check_logits_y3(logits3)
    y = _get_targets_y3(batch, y_key=y_key)

    if logits3.shape != y.shape:
        raise ValueError(f"Shape mismatch: logits3 {tuple(logits3.shape)} vs y {tuple(y.shape)}")

    # 先算每個 channel 的 mean loss，再做權重加總
    # reduction='none' -> (B,3,T) -> mean over (B,T) -> (3,)
    loss_map = F.binary_cross_entropy_with_logits(logits3, y, reduction="none")
    per_ch = loss_map.mean(dim=(0, 2))  # (3,)

    w = weights.as_tensor(device=per_ch.device, dtype=per_ch.dtype)  # (3,)
    return (w * per_ch).sum()


def bce_logits_loss_y3_tensor_with_pos_weight(
    logits3: torch.Tensor,
    batch: Mapping[str, Any],
    pos_weight: torch.Tensor,
    weights: LossWeights3 = LossWeights3(),
    y_key: str = "y",
) -> torch.Tensor:
    """
    pos_weight: shape (3,) 對應 [P,S,Event]
      - 用於 class imbalance（正類稀少時常用）
      - 注意 pos_weight 是「正類權重」，不是整體 channel 權重
    """
    _check_logits_y3(logits3)
    y = _get_targets_y3(batch, y_key=y_key)

    if logits3.shape != y.shape:
        raise ValueError(f"Shape mismatch: logits3 {tuple(logits3.shape)} vs y {tuple(y.shape)}")

    if not torch.is_tensor(pos_weight) or pos_weight.ndim != 1 or pos_weight.numel() != 3:
        raise ValueError(f"pos_weight must be tensor shape (3,), got {None if not torch.is_tensor(pos_weight) else tuple(pos_weight.shape)}")

    pw = pos_weight.to(device=logits3.device, dtype=logits3.dtype)  # (3,)
    # BCEWithLogits 的 pos_weight 會 broadcast 到 (B,3,T)
    loss_map = F.binary_cross_entropy_with_logits(logits3, y, reduction="none", pos_weight=pw)
    per_ch = loss_map.mean(dim=(0, 2))  # (3,)

    w = weights.as_tensor(device=per_ch.device, dtype=per_ch.dtype)
    return (w * per_ch).sum()


def bce_logits_loss_y3_tensor_simple(
    logits3: torch.Tensor,
    batch: Mapping[str, Any],
    y_key: str = "y",
) -> torch.Tensor:
    """
    最簡版：完全不加權（等價於對 (B,3,T) 整體 mean）
    """
    _check_logits_y3(logits3)
    y = _get_targets_y3(batch, y_key=y_key)

    if logits3.shape != y.shape:
        raise ValueError(f"Shape mismatch: logits3 {tuple(logits3.shape)} vs y {tuple(y.shape)}")

    return F.binary_cross_entropy_with_logits(logits3, y, reduction="mean")


def _as_alpha_tensor3(alpha: Sequence[float] | torch.Tensor, device, dtype) -> torch.Tensor:
    if torch.is_tensor(alpha):
        a = alpha.to(device=device, dtype=dtype)
    else:
        a = torch.tensor(list(alpha), device=device, dtype=dtype)
    if a.ndim != 1 or a.numel() != 3:
        raise ValueError(f"alpha must be length-3 for [P,S,event], got shape={tuple(a.shape)}")
    return a


def focal_logits_loss_y3_tensor(
    logits3: torch.Tensor,
    batch: Mapping[str, Any],
    weights: LossWeights3 = LossWeights3(),
    alpha: Sequence[float] | torch.Tensor = (0.25, 0.25, 0.25),
    gamma: float = 2.0,
    eps: float = 1e-6,
    y_key: str = "y",
) -> torch.Tensor:
    """
    Binary focal loss on logits for (B,3,T), with channel order [P,S,Event].

    FL = - alpha_t * (1 - p_t)^gamma * log(p_t + eps)
    where:
      p_t = y*p + (1-y)*(1-p), p=sigmoid(logits)
      alpha_t = y*alpha + (1-y)*(1-alpha)
    """
    _check_logits_y3(logits3)
    y = _get_targets_y3(batch, y_key=y_key)
    if logits3.shape != y.shape:
        raise ValueError(f"Shape mismatch: logits3 {tuple(logits3.shape)} vs y {tuple(y.shape)}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    p = torch.sigmoid(logits3)
    p_t = y * p + (1.0 - y) * (1.0 - p)  # (B,3,T)
    alpha_ch = _as_alpha_tensor3(alpha, device=logits3.device, dtype=logits3.dtype).view(1, 3, 1)
    alpha_t = y * alpha_ch + (1.0 - y) * (1.0 - alpha_ch)  # (B,3,T)

    loss_map = -alpha_t * ((1.0 - p_t) ** gamma) * torch.log(p_t + eps)  # (B,3,T)
    per_ch = loss_map.mean(dim=(0, 2))  # (3,)
    w = weights.as_tensor(device=per_ch.device, dtype=per_ch.dtype)
    return (w * per_ch).sum()
