from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor, normalize, resize, center_crop, InterpolationMode
from PIL import Image

from src.utils.io import load_config
from src.models.factory import build_model
from src.degradations import DeterministicDegrader
from src.utils.viz import mask_to_color, overlay_mask

def resize_short(img: Image.Image, short_size: int, interp):
    w, h = img.size
    if h < w:
        new_h = short_size
        new_w = int(w * short_size / h)
    else:
        new_w = short_size
        new_h = int(h * short_size / w)
    return resize(img, [new_h, new_w], interpolation=interp)

def preprocess_image(img: Image.Image, cfg):
    tcfg = cfg["data"]["transform"]
    base_size = int(tcfg.get("base_size", 520))
    crop_size = int(tcfg.get("crop_size", 480))
    mean = tcfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = tcfg.get("normalize_std", [0.229, 0.224, 0.225])

    img_r = resize_short(img, base_size, InterpolationMode.BILINEAR)
    img_r = center_crop(img_r, [crop_size, crop_size])
    x = normalize(to_tensor(img_r), mean=mean, std=std).unsqueeze(0)
    return img_r, x

def preprocess_mask(mask: Image.Image, cfg):
    tcfg = cfg["data"]["transform"]
    base_size = int(tcfg.get("base_size", 520))
    crop_size = int(tcfg.get("crop_size", 480))
    m = resize_short(mask, base_size, InterpolationMode.NEAREST)
    m = center_crop(m, [crop_size, crop_size])
    return np.array(m, dtype=np.uint8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--indices", nargs="+", type=int, required=True)
    p.add_argument("--eval_degradation", default=None, choices=[None, "gaussian_noise", "motion_blur", "jpeg"], nargs="?")
    p.add_argument("--eval_severity", type=int, default=None)
    p.add_argument("--out_dir", default="./qual_vis")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_root = args.root or cfg["data"]["root"]
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("train", {}).get("device", "cuda") == "cuda") else "cpu")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = VOCSegmentation(root=data_root, year=str(cfg["data"].get("year", "2012")), image_set=args.split, download=False)
    degrader = DeterministicDegrader(args.eval_degradation, args.eval_severity)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        img, mask = ds[idx]
        img = img.convert("RGB")
        img_deg = degrader(img)
        img_proc, x = preprocess_image(img_deg, cfg)
        mask_np = preprocess_mask(mask, cfg)

        with torch.no_grad():
            pred = model(x.to(device))["out"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        sample_dir = out_dir / f"{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        img.save(sample_dir / "orig.png")
        img_deg.save(sample_dir / "degraded.png")
        img_proc.save(sample_dir / "degraded_resized.png")
        mask_to_color(mask_np).save(sample_dir / "gt_color.png")
        mask_to_color(pred).save(sample_dir / "pred_color.png")
        overlay_mask(img_proc, mask_np).save(sample_dir / "overlay_gt.png")
        overlay_mask(img_proc, pred).save(sample_dir / "overlay_pred.png")
        print(f"Saved {sample_dir}")

if __name__ == "__main__":
    main()
