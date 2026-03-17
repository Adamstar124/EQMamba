# -*- coding: utf-8 -*-
# train.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import yaml

import torch
from torch.utils.data import Dataset, DataLoader

from models.eqmamba import EQMamba2x, EQMamba2xConfig
from training.trainer import Trainer, TrainerConfig
from training.losses import bce_logits_loss_y3_tensor, focal_logits_loss_y3_tensor, LossWeights3
from training.metrics import Y3EventMetrics, EventMetricConfig


# ----------------------------
# Minimal dummy dataset (替換用)
# ----------------------------
class DummyY3Dataset(Dataset):
    """
    這只是讓你 train.py 可以「跑得起來」的最小資料集範例。
    你之後把它換成自己的 STEAD / synth dataset 即可。

    Return:
      batch["X"] : (3,T) float32
      batch["y"] : (3,T) float32  channel=[P,S,Event]
    """
    def __init__(self, n: int = 1024, T: int = 6000, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 3, T, generator=g)
        # 假 labels：稀疏事件（僅示範）
        self.y = (torch.rand(n, 3, T, generator=g) > 0.995).float()

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"X": self.X[idx], "y": self.y[idx]}


class _ZarrAdapterDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        out = {"X": sample["waveform"], "y": sample["labels"]}
        if "source" in sample:
            out["source"] = sample["source"]
        return out


# ----------------------------
# Opt / Sched helpers
# ----------------------------
def build_optimizer(model: torch.nn.Module, lr: float, wd: float):
    # AdamW is a safe default
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, kind: str = "plateau", plateau_patience: int = 3):
    if kind == "none":
        return None
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(plateau_patience), verbose=True
        )
    if kind == "cosine":
        # 這個 scheduler 需要你提供 total epochs；這裡先留空，fit 前再包或改用 OneCycle
        raise ValueError("cosine scheduler not wired in this minimal script; use plateau or none.")
    raise ValueError(f"Unknown scheduler kind: {kind}")


def build_loss_fn(loss_cfg: Dict[str, Any]):
    w = LossWeights3(P=loss_cfg["wP"], S=loss_cfg["wS"], event=loss_cfg["wE"])
    loss_type = str(loss_cfg.get("type", "bce")).lower()

    if loss_type == "bce":
        return lambda logits3, batch: bce_logits_loss_y3_tensor(logits3, batch, weights=w)

    if loss_type == "focal":
        focal_cfg = loss_cfg.get("focal", {}) or {}
        alpha = focal_cfg.get("alpha", [0.25, 0.25, 0.25])
        gamma = float(focal_cfg.get("gamma", 2.0))
        eps = float(focal_cfg.get("eps", 1e-6))
        return lambda logits3, batch: focal_logits_loss_y3_tensor(
            logits3,
            batch,
            weights=w,
            alpha=alpha,
            gamma=gamma,
            eps=eps,
        )

    raise ValueError(f"Unknown loss.type: {loss_type}. Supported: bce, focal")


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--resume_last", action="store_true", help="Resume from save_dir/run_name_last.pt")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a dict, got {type(cfg)}")
    return cfg


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dummy_loaders(data_cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = DummyY3Dataset(n=data_cfg["train_n"], T=data_cfg["T"], seed=seed)
    val_ds = DummyY3Dataset(n=data_cfg["val_n"], T=data_cfg["T"], seed=seed + 1)
    prefetch_factor = data_cfg.get("prefetch_factor")
    persistent_workers = bool(data_cfg.get("persistent_workers", False))
    if data_cfg["num_workers"] == 0:
        prefetch_factor = None
        persistent_workers = False

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_loader, val_loader


def _resolve_zarr_paths(
    root_dir: str,
    split: str,
    events_per_sample,
    override_paths: Optional[List[str]],
):
    if override_paths:
        return override_paths
    from data.preprocess_zarr_dataset import collect_default_paths

    return collect_default_paths(root_dir, split, events_per_sample)


def _build_zarr_loaders(data_cfg: Dict[str, Any], seed: int) -> Tuple[DataLoader, DataLoader, int]:
    from data.preprocess_zarr_dataset import ZarrEventDataset

    zcfg = data_cfg["zarr"]
    root_dir = zcfg.get("root_dir", "/data/output")
    events_per_sample = zcfg.get("events_per_sample", 2)

    train_paths = _resolve_zarr_paths(
        root_dir=root_dir,
        split="train",
        events_per_sample=events_per_sample,
        override_paths=zcfg.get("train_paths"),
    )
    val_paths = _resolve_zarr_paths(
        root_dir=root_dir,
        split="val",
        events_per_sample=events_per_sample,
        override_paths=zcfg.get("val_paths"),
    )

    if not train_paths:
        raise ValueError("No Zarr train paths found. Check data.zarr.root_dir or data.zarr.train_paths.")
    if not val_paths:
        raise ValueError("No Zarr val paths found. Check data.zarr.root_dir or data.zarr.val_paths.")

    train_base = ZarrEventDataset(
        train_paths,
        mode="train",
        bandpass=tuple(zcfg.get("bandpass", [1.0, 45.0])),
        triangle_half_width=int(zcfg.get("triangle_half_width", 10)),
        gap_prob=float(zcfg.get("gap_prob", 0.2)),
        gap_sec=(float(zcfg.get("gap_min_sec", 0.5)), float(zcfg.get("gap_max_sec", 5.0))),
        drop_prob=float(zcfg.get("drop_prob", 0.3)),
        drop_noise_scale=float(zcfg.get("drop_noise_scale", 1.0)),
        seed=seed,
        use_written=bool(zcfg.get("use_written", True)),
        source_fractions=zcfg.get("source_fractions"),
        return_source=True,
    )
    val_base = ZarrEventDataset(
        val_paths,
        mode="eval",
        bandpass=tuple(zcfg.get("bandpass", [1.0, 45.0])),
        triangle_half_width=int(zcfg.get("triangle_half_width", 10)),
        gap_prob=0.0,
        gap_sec=(float(zcfg.get("gap_min_sec", 0.5)), float(zcfg.get("gap_max_sec", 5.0))),
        drop_prob=0.0,
        drop_noise_scale=float(zcfg.get("drop_noise_scale", 1.0)),
        seed=seed + 1,
        use_written=bool(zcfg.get("use_written", True)),
        source_fractions=zcfg.get("source_fractions"),
        return_source=True,
    )

    train_ds = _ZarrAdapterDataset(train_base)
    val_ds = _ZarrAdapterDataset(val_base)

    sample = train_ds[0]
    inferred_T = int(sample["X"].shape[-1])
    prefetch_factor = data_cfg.get("prefetch_factor")
    persistent_workers = bool(data_cfg.get("persistent_workers", False))
    if data_cfg["num_workers"] == 0:
        prefetch_factor = None
        persistent_workers = False

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_loader, val_loader, inferred_T


def main():
    args = parse_args()
    cfg = load_config(args.config)

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    metric_cfg = cfg["metrics"]
    model_cfg = cfg["model"]
    cnn_cfg = model_cfg["cnn"]
    down_k = cnn_cfg.get("down_k")
    down_p = cnn_cfg.get("down_p")

    set_seed(run_cfg["seed"])

    # 1) Data
    data_source = data_cfg.get("source", "dummy")
    if data_source == "dummy":
        train_loader, val_loader = _build_dummy_loaders(data_cfg, run_cfg["seed"])
    elif data_source == "zarr":
        train_loader, val_loader, inferred_T = _build_zarr_loaders(data_cfg, run_cfg["seed"])
        if "T" in data_cfg and data_cfg["T"] is not None:
            if int(data_cfg["T"]) != int(inferred_T):
                raise ValueError(f"Config T={data_cfg['T']} does not match dataset length={inferred_T}.")
        else:
            data_cfg["T"] = int(inferred_T)
    else:
        raise ValueError(f"Unknown data.source: {data_source}")

    # 2) Model config
    model_cfg = EQMamba2xConfig(
        in_samples=data_cfg["T"],
        d_model=model_cfg["d_model"],
        d_state=model_cfg["d_state"],
        core_mamba_nums=model_cfg["core_mamba_nums"],
        decode_mamba_nums=model_cfg["decode_mamba_nums"],
        performer_heads=model_cfg["performer_heads"],
        cnn=EQMamba2xConfig.cnn.__class__(  # StridedCNN2xConfig
            in_ch=3,
            k=cnn_cfg["k"],
            down_k=None if down_k is None else int(down_k),
            down_p=None if down_p is None else int(down_p),
            skip_refine_k1=bool(cnn_cfg.get("skip_refine_k1", False)),
            norm=cnn_cfg["norm"],
            act=cnn_cfg["act"],
            dropout=0.0,
            chs=(cnn_cfg["c0"], cnn_cfg["c1"], cnn_cfg["c2"]),
        ),
        # head_names 固定 [P,S,event]（你在 config 裡已經預設）
    )

    # safety: ensure d_model == c2
    if model_cfg.d_model != model_cfg.cnn.chs[2]:
        raise ValueError(f"d_model must equal cnn bottleneck channels c2. Got d_model={model_cfg.d_model}, c2={model_cfg.cnn.chs[2]}")

    model = EQMamba2x(model_cfg)

    # 3) Optimizer / Scheduler
    optimizer = build_optimizer(model, lr=train_cfg["lr"], wd=train_cfg["wd"])
    scheduler = build_scheduler(
        optimizer,
        kind=train_cfg["sched"],
        plateau_patience=train_cfg.get("sched_patience", 3),
    )

    # 4) Loss / Metrics
    loss_fn = build_loss_fn(loss_cfg)

    metrics = Y3EventMetrics(
        EventMetricConfig(
            thr_p=metric_cfg["thr_p"],
            thr_s=metric_cfg["thr_s"],
            thr_event=metric_cfg["thr_event"],
            tol_p=metric_cfg["tol_p"],
            tol_s=metric_cfg["tol_s"],
            iou_thr=metric_cfg["iou_thr"],
            merge_gap_event=metric_cfg["merge_gap_event"],
        )
    )

    # 5) Trainer
    trainer_cfg = TrainerConfig(
        device=run_cfg["device"],
        amp=train_cfg["amp"],
        grad_clip_norm=train_cfg["grad_clip"],
        log_every=train_cfg["log_every"],
        log_csv=train_cfg.get("log_csv", True),
        csv_path=train_cfg.get("csv_path"),
        early_stop_patience=train_cfg.get("early_stop_patience", 0),
        save_dir=run_cfg["save_dir"],
        run_name=run_cfg["run_name"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        cfg=trainer_cfg,
        metrics=metrics,
    )

    # 6) Optionally resume from last checkpoint
    start_epoch = 0
    if args.resume_last:
        ckpt_path = Path(run_cfg["save_dir"]) / f"{run_cfg['run_name']}_last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume requested but checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=trainer.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        if ckpt.get("sched") is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["sched"])
        if ckpt.get("scaler") is not None:
            trainer.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("best_val") is not None:
            trainer.best_val = float(ckpt["best_val"])
        if ckpt.get("epoch") is not None:
            start_epoch = int(ckpt["epoch"])
        print(f"Resumed from: {ckpt_path} (start_epoch={start_epoch})")

    # 7) Save run config snapshot (optional but useful)
    Path(run_cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(run_cfg["save_dir"]) / f"{run_cfg['run_name']}_config.pt"
    torch.save(
        {"config": cfg, "model_cfg": asdict(model_cfg), "trainer_cfg": asdict(trainer_cfg)},
        snapshot_path,
    )
    print(f"Saved config snapshot to: {snapshot_path}")

    # 8) Fit
    trainer.fit(train_loader, val_loader, epochs=train_cfg["epochs"], start_epoch=start_epoch)
    print("Training finished.")


if __name__ == "__main__":
    main()
