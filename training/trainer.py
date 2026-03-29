# eqmamba/training/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List
import copy

import csv
import torch
from torch import nn
from tqdm import tqdm

from .utils import move_to_device


@dataclass
class TrainerConfig:
    device: str = "cuda"
    amp: bool = True
    grad_clip_norm: float = 1.0
    log_every: int = 50
    log_csv: bool = True
    csv_path: Optional[str] = None
    early_stop_patience: int = 0

    save_dir: str = "./checkpoints"
    run_name: str = "eqmamba_run"


class Trainer:
    """
    Tensor-only 版本 Trainer（配合你現在的介面）：

    - model(batch["X"]) -> logits3: (B,3,T)
    - batch["y"]        -> y3:      (B,3,T)  channel=[P,S,Event]
    - loss_fn(logits3, batch) -> scalar loss
    - metrics.update(logits3, batch["y"]) （可選）
    - batch["source"] (optional) -> per-source stats/metrics

    這版不再處理 dict logits，也不再拆 channel 成三份，整條 pipeline 更單純。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        cfg: TrainerConfig,
        loss_items_fn: Optional[Callable[[torch.Tensor, Dict[str, Any]], Dict[str, float]]] = None,
        metrics: Optional[Any] = None,  # e.g. Y3EventMetrics
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.loss_items_fn = loss_items_fn
        self.cfg = cfg
        self.metrics = metrics

        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and self.device.type == "cuda"))

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        self.csv_path = (
            Path(cfg.csv_path)
            if cfg.csv_path
            else Path(cfg.save_dir) / f"{cfg.run_name}_metrics.csv"
        )
        self.best_val = float("inf")

    def _scheduler_step(self, val_loss: float):
        if self.scheduler is None:
            return
        if "ReduceLROnPlateau" in self.scheduler.__class__.__name__:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def save(self, name: str, epoch: Optional[int] = None):
        path = Path(self.cfg.save_dir) / f"{self.cfg.run_name}_{name}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.optimizer.state_dict(),
                "sched": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "cfg": self.cfg.__dict__,
                "epoch": int(epoch) if epoch is not None else None,
                "best_val": float(self.best_val),
            },
            path,
        )

    @staticmethod
    def _loss_to_float(loss: torch.Tensor) -> float:
        return float(loss.detach().item())

    @staticmethod
    def _current_lr(optimizer: torch.optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _clone_metrics(self):
        if self.metrics is None:
            return None
        cloned = copy.deepcopy(self.metrics)
        cloned.reset()
        return cloned

    @staticmethod
    def _group_indices(sources: List[Any]) -> Dict[str, List[int]]:
        grouped: Dict[str, List[int]] = {}
        for i, src in enumerate(sources):
            key = str(src)
            grouped.setdefault(key, []).append(i)
        return grouped

    def _init_source_stats(self):
        return {"loss_sum": 0.0, "count": 0, "metrics": self._clone_metrics()}

    @staticmethod
    def _source_stats_to_row(source_stats: Dict[str, Dict[str, Any]], prefix: str) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for src, stats in source_stats.items():
            count = stats.get("count", 0)
            avg_loss = stats.get("loss_sum", 0.0) / max(count, 1)
            row[f"{prefix}/{src}/loss"] = float(avg_loss)
            metrics = stats.get("metrics")
            if metrics is not None:
                row.update(metrics.as_dict(prefix=f"{prefix}/{src}"))
        return row

    def train_one_epoch(self, train_loader, epoch: int, compute_metrics: bool = True, compute_source_stats: bool = True):
        self.model.train()
        if compute_metrics and self.metrics is not None:
            self.metrics.reset()

        loss_sum = 0.0
        loss_items_sum: Dict[str, float] = {}
        step = 0
        source_stats: Dict[str, Dict[str, Any]] = {} if compute_source_stats else {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False)
        for batch in pbar:
            batch = move_to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                logits3 = self.model(batch["X"])          # (B,3,T)
                loss = self.loss_fn(logits3, batch)       # 由 loss_fn 從 batch["y"] 取 target

            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_f = self._loss_to_float(loss)
            loss_sum += loss_f
            step += 1
            pbar.set_postfix(loss=f"{loss_f:.4f}", lr=f"{self._current_lr(self.optimizer):.6g}")

            if self.loss_items_fn is not None:
                with torch.no_grad():
                    items = self.loss_items_fn(logits3.detach(), {"y": batch["y"].detach()})
                    for k, v in items.items():
                        loss_items_sum[k] = loss_items_sum.get(k, 0.0) + float(v)

            if compute_metrics and self.metrics is not None:
                # ✅ tensor-only metrics
                self.metrics.update(logits3, batch["y"])

            if compute_source_stats and "source" in batch:
                sources = batch["source"]
                grouped = self._group_indices(sources)
                with torch.no_grad():
                    logits_det = logits3.detach()
                    y_det = batch["y"].detach()
                    for src, idxs in grouped.items():
                        stats = source_stats.setdefault(src, self._init_source_stats())
                        logits_s = logits_det[idxs]
                        y_s = y_det[idxs]
                        loss_s = self.loss_fn(logits_s, {"y": y_s})
                        stats["loss_sum"] += float(loss_s.detach().item())
                        stats["count"] += 1
                        if stats["metrics"] is not None:
                            stats["metrics"].update(logits_s, y_s)

        loss_items_avg = {k: (v / max(step, 1)) for k, v in loss_items_sum.items()}
        return loss_sum / max(step, 1), loss_items_avg, source_stats

    @torch.no_grad()
    def validate(self, val_loader, epoch: int, compute_metrics: bool = True, compute_source_stats: bool = True):
        self.model.eval()
        if compute_metrics and self.metrics is not None:
            self.metrics.reset()

        loss_sum = 0.0
        loss_items_sum: Dict[str, float] = {}
        step = 0
        source_stats: Dict[str, Dict[str, Any]] = {} if compute_source_stats else {}

        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [val]", leave=False)
        for batch in pbar:
            batch = move_to_device(batch, self.device)

            logits3 = self.model(batch["X"])            # (B,3,T)
            loss = self.loss_fn(logits3, batch)

            loss_f = self._loss_to_float(loss)
            loss_sum += loss_f
            step += 1
            pbar.set_postfix(loss=f"{loss_f:.4f}")

            if self.loss_items_fn is not None:
                items = self.loss_items_fn(logits3.detach(), {"y": batch["y"].detach()})
                for k, v in items.items():
                    loss_items_sum[k] = loss_items_sum.get(k, 0.0) + float(v)

            if compute_metrics and self.metrics is not None:
                self.metrics.update(logits3, batch["y"])

            if compute_source_stats and "source" in batch:
                sources = batch["source"]
                grouped = self._group_indices(sources)
                logits_det = logits3.detach()
                y_det = batch["y"].detach()
                for src, idxs in grouped.items():
                    stats = source_stats.setdefault(src, self._init_source_stats())
                    logits_s = logits_det[idxs]
                    y_s = y_det[idxs]
                    loss_s = self.loss_fn(logits_s, {"y": y_s})
                    stats["loss_sum"] += float(loss_s.detach().item())
                    stats["count"] += 1
                    if stats["metrics"] is not None:
                        stats["metrics"].update(logits_s, y_s)

        loss_items_avg = {k: (v / max(step, 1)) for k, v in loss_items_sum.items()}
        return loss_sum / max(step, 1), loss_items_avg, source_stats

    def fit(self, train_loader, val_loader, epochs: int, start_epoch: int = 0):
        no_improve = 0
        patience = int(self.cfg.early_stop_patience or 0)
        log_interval = max(1, int(self.cfg.log_every or 1))
        for epoch in range(start_epoch, epochs):
            should_log = ((epoch + 1) % log_interval == 0) or (epoch == epochs - 1)
            tr, tr_loss_items, tr_sources = self.train_one_epoch(
                train_loader, epoch, compute_metrics=should_log, compute_source_stats=should_log
            )
            train_metrics = (
                self.metrics.as_dict(prefix="train") if (self.metrics is not None and should_log) else {}
            )
            va, va_loss_items, va_sources = self.validate(
                val_loader, epoch, compute_metrics=should_log, compute_source_stats=should_log
            )
            val_metrics = (
                self.metrics.as_dict(prefix="val") if (self.metrics is not None and should_log) else {}
            )

            if va < self.best_val:
                self.best_val = va
                self.save("best", epoch=epoch + 1)
                no_improve = 0
            else:
                if patience > 0:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping: no improvement for {no_improve} epoch(s).")
                        break

            self._scheduler_step(va)

            if self.cfg.log_csv:
                row = {
                    "epoch": int(epoch + 1),
                    "train/loss": float(tr),
                    "val/loss": float(va),
                    "lr": self._current_lr(self.optimizer),
                }
                row.update({f"train/{k}": float(v) for k, v in tr_loss_items.items()})
                row.update({f"val/{k}": float(v) for k, v in va_loss_items.items()})
                if should_log:
                    row.update(train_metrics)
                    row.update(val_metrics)
                    row.update(self._source_stats_to_row(tr_sources, prefix="train"))
                    row.update(self._source_stats_to_row(va_sources, prefix="val"))

                if self.csv_path.exists():
                    with self.csv_path.open("r", newline="") as f:
                        reader = csv.DictReader(f)
                        header = reader.fieldnames
                        existing_rows = list(reader)
                else:
                    header = None

                base_keys = ["epoch", "train/loss", "val/loss", "lr"]
                if not header:
                    metric_keys = sorted(k for k in row.keys() if k not in base_keys)
                    header = base_keys + metric_keys
                    with self.csv_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerow(row)
                else:
                    missing = [k for k in row.keys() if k not in header]
                    if missing:
                        all_keys = set(header) | set(row.keys())
                        metric_keys = sorted(k for k in all_keys if k not in base_keys)
                        header = base_keys + metric_keys
                        with self.csv_path.open("w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                            writer.writeheader()
                            for old in existing_rows:
                                writer.writerow(old)
                    with self.csv_path.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                        writer.writerow(row)

            # Always update last checkpoint each epoch for resumability.
            self.save("last", epoch=epoch + 1)

        # Final safety save (covers zero-epoch case).
        self.save("last", epoch=epoch + 1 if "epoch" in locals() else None)
