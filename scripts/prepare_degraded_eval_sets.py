"""
可选脚本：把 VOC val 预生成成不同退化版本（离线图片），便于人工浏览/展示。
训练与评估不依赖该脚本，因为 src.evaluate 支持在线退化。
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
from torchvision.datasets import VOCSegmentation
from src.degradations import apply_degradation

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data")
    p.add_argument("--year", default="2012")
    p.add_argument("--split", default="val")
    p.add_argument("--out_root", default="./degraded_eval_preview")
    p.add_argument("--types", nargs="+", default=["gaussian_noise", "motion_blur", "jpeg"])
    p.add_argument("--severities", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--max_samples", type=int, default=50)
    args = p.parse_args()

    ds = VOCSegmentation(root=args.root, year=args.year, image_set=args.split, download=False)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.max_samples, len(ds))):
        img, _ = ds[i]
        img = img.convert("RGB")
        for t in args.types:
            for s in args.severities:
                out = apply_degradation(img, t, s)
                d = out_root / t / f"s{s}"
                d.mkdir(parents=True, exist_ok=True)
                out.save(d / f"{i:05d}.jpg", quality=95)
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1} samples")
    print("Done.")

if __name__ == "__main__":
    main()
