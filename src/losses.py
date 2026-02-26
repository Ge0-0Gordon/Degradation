from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W]
    device = x.device
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=x.dtype, device=device)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=x.dtype, device=device)
    B, C, H, W = x.shape
    x_ = x.reshape(B * C, 1, H, W)
    gx = F.conv2d(x_, kx, padding=1)
    gy = F.conv2d(x_, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return g.reshape(B, C, H, W)

class SegCriterion(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        lcfg = cfg.get("loss", {})
        self.ignore_index = int(lcfg.get("ignore_index", 255))
        self.ce_weight = float(lcfg.get("ce_weight", 1.0))
        self.aux_weight = float(lcfg.get("aux_weight", 0.4))
        self.boundary_weight = float(lcfg.get("boundary_weight", 0.0))
        self.num_classes = int(cfg["model"].get("num_classes", 21))
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def boundary_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        tgt = target.clone()
        tgt[tgt == self.ignore_index] = 0
        onehot = F.one_hot(tgt.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = (target != self.ignore_index).unsqueeze(1).float()
        onehot = onehot * valid
        pred_edges = _sobel_edges(probs)
        gt_edges = _sobel_edges(onehot)
        return F.l1_loss(pred_edges * valid, gt_edges * valid)

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor):
        loss_ce = self.ce(outputs["out"], target)
        total = self.ce_weight * loss_ce
        logs = {"loss_ce": float(loss_ce.detach().cpu())}
        if "aux" in outputs and outputs["aux"] is not None:
            loss_aux = self.ce(outputs["aux"], target)
            total = total + self.aux_weight * loss_aux
            logs["loss_aux"] = float(loss_aux.detach().cpu())
        if self.boundary_weight > 0:
            loss_b = self.boundary_loss(outputs["out"], target)
            total = total + self.boundary_weight * loss_b
            logs["loss_boundary"] = float(loss_b.detach().cpu())
        logs["loss_total"] = float(total.detach().cpu())
        return total, logs
