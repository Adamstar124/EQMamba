# eqmamba/training/utils.py
from typing import Any, Dict
import torch

def move_to_device(batch: Dict[str, Any], device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
