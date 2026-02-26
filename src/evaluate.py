from __future__ import annotations
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.voc_seg import VOCSegDataset
from src.models.factory import build_model
from src.losses import SegCriterion
from src.utils.io import load_config, ensure_dir
from src.utils.metrics import SegmentationMetric
from src.utils.seed import set_seed

@torch.no_grad()
def run_eval(model, loader, device, criterion, num_classes=21, ignore_index=255):
    model.eval()
    metric = SegmentationMetric(num_classes, ignore_index)
    losses = []
    for images, targets, _ in tqdm(loader, desc="Eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss, _ = criterion(outputs, targets)
        losses.append(float(loss.detach().cpu()))
        preds = outputs["out"].argmax(dim=1)
        metric.update(preds, targets)
    stats = metric.compute()
    stats["loss"] = float(sum(losses) / max(1, len(losses)))
    return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--eval_degradation", default=None, choices=[None, "gaussian_noise", "motion_blur", "jpeg"], nargs="?")
    p.add_argument("--eval_severity", type=int, default=None)
    p.add_argument("--save_dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("train", {}).get("device", "cuda") == "cuda") else "cpu")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    dcfg = cfg["data"]
    ds = VOCSegDataset(
        root=dcfg["root"],
        image_set=args.split,
        year=str(dcfg.get("year", "2012")),
        download=False,
        train=False,
        transform_cfg=dcfg.get("transform", {}),
        degradation_cfg={"enable": False},
        eval_degradation=args.eval_degradation,
        eval_severity=args.eval_severity,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("val_batch_size", cfg["train"].get("batch_size", 8))),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
    )
    criterion = SegCriterion(cfg)
    stats = run_eval(
        model, loader, device, criterion,
        num_classes=int(cfg["model"].get("num_classes", 21)),
        ignore_index=int(cfg.get("loss", {}).get("ignore_index", 255)),
    )
    stats.update({
        "split": args.split,
        "eval_degradation": args.eval_degradation,
        "eval_severity": args.eval_severity,
        "ckpt": args.ckpt,
        "config": args.config,
    })
    print(json.dumps(stats, indent=2))

    if args.save_dir:
        ensure_dir(args.save_dir)
        suffix = f"{args.eval_degradation or 'clean'}_s{args.eval_severity if args.eval_severity is not None else 0}"
        with open(os.path.join(args.save_dir, f"eval_{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
