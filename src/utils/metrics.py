from __future__ import annotations
import numpy as np
import torch

class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confmat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_np = preds.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        for p, t in zip(preds_np, targets_np):
            mask = t != self.ignore_index
            p = p[mask]
            t = t[mask]
            if p.size == 0:
                continue
            idx = self.num_classes * t + p
            binc = np.bincount(idx, minlength=self.num_classes ** 2)
            self.confmat += binc.reshape(self.num_classes, self.num_classes)

    def compute(self):
        h = self.confmat
        acc = np.diag(h).sum() / (h.sum() + 1e-12)
        acc_cls = np.diag(h) / (h.sum(axis=1) + 1e-12)
        iu = np.diag(h) / (h.sum(axis=1) + h.sum(axis=0) - np.diag(h) + 1e-12)
        freq = h.sum(axis=1) / (h.sum() + 1e-12)
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()
        return {
            "pixel_acc": float(acc),
            "mean_acc_cls": float(np.nanmean(acc_cls)),
            "miou": float(np.nanmean(iu)),
            "fw_iou": float(fw_iou),
            "iou_per_class": iu.tolist(),
        }
