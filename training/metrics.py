# eqmamba/training/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class EventMetricConfig:
    """
    Event-level metrics config for P/S peaks and Event intervals.
    """
    thr_p: float = 0.5
    thr_s: float = 0.5
    thr_event: float = 0.5
    tol_p: int = 10          # in samples
    tol_s: int = 10          # in samples
    iou_thr: float = 0.3
    merge_gap_event: int = 0
    eps: float = 1e-9


class EventConfusionMeter:
    """
    累積 event-level confusion matrix：TP/TN/FP/FN
    TN 以「訊號級」計數：該訊號無 GT 且無預測才算 1 個 TN
    """
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, tp: int, fp: int, fn: int, tn: int):
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}/TP": float(self.tp),
            f"{prefix}/TN": float(self.tn),
            f"{prefix}/FP": float(self.fp),
            f"{prefix}/FN": float(self.fn),
        }

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0


class Y3EventMetrics:
    """
    Event-level metrics:
      - P/S: peak matching with tolerance
      - Event: interval IoU matching
      - TN: per-signal when no GT and no prediction
    """
    def __init__(self, cfg: EventMetricConfig = EventMetricConfig()):
        self.cfg = cfg
        self._meters = {
            "P": EventConfusionMeter(),
            "S": EventConfusionMeter(),
            "event": EventConfusionMeter(),
        }
        self._event_iou_sum = 0.0
        self._event_iou_cnt = 0

    @staticmethod
    def _check_y3(x: torch.Tensor, name: str):
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")
        if x.ndim != 3 or x.size(1) != 3:
            raise ValueError(f"{name} must have shape (B,3,T), got {tuple(x.shape)}")

    def reset(self):
        for m in self._meters.values():
            m.reset()
        self._event_iou_sum = 0.0
        self._event_iou_cnt = 0

    @staticmethod
    def _segments_above(x: torch.Tensor, thr: float) -> List[Tuple[int, int]]:
        """
        Return list of (start, end) inclusive where x >= thr.
        """
        x = x.detach().cpu().float()
        above = (x >= thr).numpy().tolist()
        segments: List[Tuple[int, int]] = []
        start = None
        for i, flag in enumerate(above):
            if flag and start is None:
                start = i
            if (not flag or i == len(above) - 1) and start is not None:
                end = i if flag and i == len(above) - 1 else i - 1
                segments.append((start, end))
                start = None
        return segments

    @staticmethod
    def _peaks_from_segments(x: torch.Tensor, thr: float) -> List[int]:
        """
        Convert thresholded segments into peak indices (argmax per segment).
        """
        segments = Y3EventMetrics._segments_above(x, thr)
        peaks: List[int] = []
        for s, e in segments:
            seg = x[s : e + 1]
            if seg.numel() == 0:
                continue
            rel = int(torch.argmax(seg).item())
            peaks.append(s + rel)
        return peaks

    @staticmethod
    def _match_peaks(pred: List[int], gt: List[int], tol: int) -> Tuple[int, int, int]:
        """
        Greedy one-to-one match by nearest GT within tolerance.
        Returns tp, fp, fn.
        """
        if not pred and not gt:
            return 0, 0, 0
        pred_used = [False] * len(pred)
        tp = 0
        for g in gt:
            best_j = -1
            best_d = None
            for j, p in enumerate(pred):
                if pred_used[j]:
                    continue
                d = abs(p - g)
                if d <= tol and (best_d is None or d < best_d):
                    best_d = d
                    best_j = j
            if best_j >= 0:
                pred_used[best_j] = True
                tp += 1
        fp = pred_used.count(False)
        fn = len(gt) - tp
        return tp, fp, fn

    @staticmethod
    def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        a0, a1 = a
        b0, b1 = b
        inter = max(0, min(a1, b1) - max(a0, b0) + 1)
        if inter == 0:
            return 0.0
        union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
        return float(inter) / float(union)

    def _match_intervals(
        self,
        pred: List[Tuple[int, int]],
        gt: List[Tuple[int, int]],
    ) -> Tuple[int, int, int, float, int]:
        """
        One-to-one match by max IoU with threshold.
        Returns tp, fp, fn, iou_sum, iou_cnt.
        """
        if not pred and not gt:
            return 0, 0, 0, 0.0, 0
        pairs: List[Tuple[float, int, int]] = []
        for gi, g in enumerate(gt):
            for pi, p in enumerate(pred):
                iou = self._iou(p, g)
                if iou >= self.cfg.iou_thr:
                    pairs.append((iou, gi, pi))
        pairs.sort(reverse=True, key=lambda x: x[0])
        gt_used = [False] * len(gt)
        pred_used = [False] * len(pred)
        tp = 0
        iou_sum = 0.0
        iou_cnt = 0
        for iou, gi, pi in pairs:
            if gt_used[gi] or pred_used[pi]:
                continue
            gt_used[gi] = True
            pred_used[pi] = True
            tp += 1
            iou_sum += iou
            iou_cnt += 1
        fp = pred_used.count(False)
        fn = gt_used.count(False)
        return tp, fp, fn, iou_sum, iou_cnt

    @torch.no_grad()
    def update(self, logits3: torch.Tensor, y3: torch.Tensor):
        """
        logits3: (B,3,T)
        y3: (B,3,T)  0/1 or soft labels
        """
        self._check_y3(logits3, "logits3")
        self._check_y3(y3, "y3")

        if logits3.shape != y3.shape:
            raise ValueError(f"Shape mismatch: logits3 {tuple(logits3.shape)} vs y3 {tuple(y3.shape)}")

        probs = torch.sigmoid(logits3)
        B = probs.size(0)
        for b in range(B):
            # P peaks
            pred_p = self._peaks_from_segments(probs[b, 0], self.cfg.thr_p)
            gt_p = self._peaks_from_segments(y3[b, 0], 0.5)
            tp, fp, fn = self._match_peaks(pred_p, gt_p, self.cfg.tol_p)
            tn = 1 if not pred_p and not gt_p else 0
            self._meters["P"].add(tp, fp, fn, tn)

            # S peaks
            pred_s = self._peaks_from_segments(probs[b, 1], self.cfg.thr_s)
            gt_s = self._peaks_from_segments(y3[b, 1], 0.5)
            tp, fp, fn = self._match_peaks(pred_s, gt_s, self.cfg.tol_s)
            tn = 1 if not pred_s and not gt_s else 0
            self._meters["S"].add(tp, fp, fn, tn)

            # Event intervals
            pred_event = self._segments_above(probs[b, 2], self.cfg.thr_event)
            gt_event = self._segments_above(y3[b, 2], 0.5)
            if self.cfg.merge_gap_event > 0 and pred_event:
                merged: List[Tuple[int, int]] = []
                cur_s, cur_e = pred_event[0]
                for s, e in pred_event[1:]:
                    if s - cur_e - 1 <= self.cfg.merge_gap_event:
                        cur_e = e
                    else:
                        merged.append((cur_s, cur_e))
                        cur_s, cur_e = s, e
                merged.append((cur_s, cur_e))
                pred_event = merged
            tp, fp, fn, iou_sum, iou_cnt = self._match_intervals(pred_event, gt_event)
            tn = 1 if not pred_event and not gt_event else 0
            self._meters["event"].add(tp, fp, fn, tn)
            self._event_iou_sum += iou_sum
            self._event_iou_cnt += iou_cnt

    def _stats_from_meter(self, m: EventConfusionMeter) -> Dict[str, float]:
        tp, tn, fp, fn = m.tp, m.tn, m.fp, m.fn
        eps = self.cfg.eps
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "acc": float(acc),
        }

    def as_dict(self, prefix: str = "val") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, m in self._meters.items():
            stats = self._stats_from_meter(m)
            out.update({
                f"{prefix}/{name}/precision": stats["precision"],
                f"{prefix}/{name}/recall": stats["recall"],
                f"{prefix}/{name}/f1": stats["f1"],
                f"{prefix}/{name}/acc": stats["acc"],
            })
            out.update(m.as_dict(prefix=f"{prefix}/{name}"))
        if self._event_iou_cnt > 0:
            out[f"{prefix}/event/iou_mean"] = float(self._event_iou_sum / self._event_iou_cnt)
        else:
            out[f"{prefix}/event/iou_mean"] = 0.0
        return out
