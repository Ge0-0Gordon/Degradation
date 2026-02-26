from __future__ import annotations
import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.voc_seg import VOCSegDataset
from src.models.factory import build_model
from src.losses import SegCriterion
from src.utils.io import load_config, ensure_dir, append_jsonl, save_json
from src.utils.metrics import SegmentationMetric
from src.utils.seed import set_seed

def get_device(cfg):
    return torch.device("cuda" if (torch.cuda.is_available() and cfg.get("train", {}).get("device", "cuda") == "cuda") else "cpu")

def build_loaders(cfg: Dict):
    dcfg = cfg["data"]
    common = dict(
        root=dcfg["root"],
        year=str(dcfg.get("year", "2012")),
        download=bool(dcfg.get("download", False)),
        transform_cfg=dcfg.get("transform", {}),
    )
    train_set = VOCSegDataset(
        image_set=dcfg.get("train_split", "train"),
        train=True,
        degradation_cfg=cfg.get("degradation", {}),
        **common,
    )
    val_set = VOCSegDataset(
        image_set=dcfg.get("val_split", "val"),
        train=False,
        degradation_cfg={"enable": False},
        **common,
    )
    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_set,
        batch_size=int(tcfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(tcfg.get("val_batch_size", tcfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
    )
    return train_loader, val_loader

def make_optimizer(cfg: Dict, model):
    ocfg = cfg.get("optim", {})
    name = str(ocfg.get("name", "adamw")).lower()
    lr = float(ocfg.get("lr", 1e-4))
    wd = float(ocfg.get("weight_decay", 1e-4))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(ocfg.get("momentum", 0.9)), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def make_scheduler(cfg: Dict, optimizer):
    scfg = cfg.get("scheduler", {"name": "poly"})
    name = str(scfg.get("name", "poly")).lower()
    if name == "none":
        return None
    if name == "poly":
        total_epochs = int(cfg["train"].get("epochs", 50))
        power = float(scfg.get("power", 0.9))
        def lr_lambda(epoch):
            return (1 - epoch / max(total_epochs, 1)) ** power
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["train"].get("epochs", 50)))
    return None

@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes=21, ignore_index=255):
    model.eval()
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
    losses = []
    for images, targets, _ in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss, _ = criterion(outputs, targets)
        losses.append(float(loss.detach().cpu()))
        preds = outputs["out"].argmax(dim=1)
        metric.update(preds, targets)
    stats = metric.compute()
    stats["val_loss"] = float(sum(losses) / max(1, len(losses)))
    return stats

def train_one_epoch(model, loader, device, optimizer, criterion, scaler, amp: bool):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_amp = amp and device.type == "cuda"
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n += 1
        pbar.set_postfix(loss=f"{total_loss/max(1,n):.4f}")
    return total_loss / max(1, n)

def save_ckpt(path, model, optimizer, scheduler, epoch, best_miou, cfg):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_miou": best_miou,
        "config": cfg,
    }, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_name", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    run_name = args.run_name or cfg.get("run_name") or Path(args.config).stem
    out_dir = os.path.join(cfg.get("output_root", "runs"), run_name)
    ensure_dir(out_dir)

    device = get_device(cfg)
    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    criterion = SegCriterion(cfg)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    amp = bool(cfg.get("train", {}).get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    epochs = int(cfg["train"].get("epochs", 50))
    eval_every = int(cfg["train"].get("eval_every", 1))
    num_classes = int(cfg["model"].get("num_classes", 21))
    ignore_index = int(cfg.get("loss", {}).get("ignore_index", 255))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "miou", "pixel_acc", "lr", "time_sec"])

    save_json(cfg, os.path.join(out_dir, "config_snapshot.json"))
    best_miou = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, scaler, amp)
        val_stats = {"miou": None, "pixel_acc": None, "val_loss": None}
        if (epoch % eval_every == 0) or (epoch == epochs):
            val_stats = evaluate(model, val_loader, device, criterion, num_classes, ignore_index)
            if val_stats["miou"] > best_miou:
                best_miou = val_stats["miou"]
                save_ckpt(best_path, model, optimizer, scheduler, epoch, best_miou, cfg)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_stats.get("val_loss"), val_stats.get("miou"), val_stats.get("pixel_acc"), lr, dt])

        append_jsonl({
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_stats,
            "lr": lr,
            "time_sec": dt,
            "best_miou_so_far": best_miou,
        }, os.path.join(out_dir, "train_log.jsonl"))

        save_ckpt(last_path, model, optimizer, scheduler, epoch, best_miou, cfg)
        print(f"[{epoch}/{epochs}] train={train_loss:.4f} val_miou={val_stats.get('miou')} best={best_miou:.4f} lr={lr:.2e}")

    print(f"Finished. Best checkpoint saved to: {best_path}")

if __name__ == "__main__":
    main()
