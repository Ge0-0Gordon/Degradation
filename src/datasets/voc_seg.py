from __future__ import annotations
import random
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image

from src.degradations import StochasticDegrader, DeterministicDegrader

class SegPairTransform:
    """对图像和 mask 执行一致的几何变换；mask 使用最近邻插值。"""
    def __init__(self, cfg: Dict, train: bool = True):
        self.train = train
        self.base_size = int(cfg.get("base_size", 520))
        self.crop_size = int(cfg.get("crop_size", 480))
        self.hflip_prob = float(cfg.get("hflip_prob", 0.5))
        self.scale_min = float(cfg.get("scale_min", 0.5))
        self.scale_max = float(cfg.get("scale_max", 2.0))
        self.normalize_mean = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
        self.normalize_std = cfg.get("normalize_std", [0.229, 0.224, 0.225])

    def _resize_short(self, img: Image.Image, mask: Image.Image, short_size: int):
        w, h = img.size
        if h < w:
            new_h = short_size
            new_w = int(w * short_size / h)
        else:
            new_w = short_size
            new_h = int(h * short_size / w)
        img = TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
        return img, mask

    def _pad_if_needed(self, img: Image.Image, mask: Image.Image, crop_size: int):
        w, h = img.size
        pad_w = max(crop_size - w, 0)
        pad_h = max(crop_size - h, 0)
        if pad_w > 0 or pad_h > 0:
            img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=255)
        return img, mask

    def _random_crop(self, img: Image.Image, mask: Image.Image, crop_size: int):
        img, mask = self._pad_if_needed(img, mask, crop_size)
        w, h = img.size
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        img = TF.crop(img, top, left, crop_size, crop_size)
        mask = TF.crop(mask, top, left, crop_size, crop_size)
        return img, mask

    def _center_crop(self, img: Image.Image, mask: Image.Image, crop_size: int):
        img, mask = self._pad_if_needed(img, mask, crop_size)
        img = TF.center_crop(img, [crop_size, crop_size])
        mask = TF.center_crop(mask, [crop_size, crop_size])
        return img, mask

    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.train:
            scale = random.uniform(self.scale_min, self.scale_max)
            short_size = max(128, int(self.base_size * scale))
            img, mask = self._resize_short(img, mask, short_size)
            if random.random() < self.hflip_prob:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            img, mask = self._random_crop(img, mask, self.crop_size)
        else:
            img, mask = self._resize_short(img, mask, self.base_size)
            img, mask = self._center_crop(img, mask, self.crop_size)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=self.normalize_mean, std=self.normalize_std)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_t, mask_t

class VOCSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        year: str = "2012",
        download: bool = False,
        train: bool = True,
        transform_cfg: Optional[Dict] = None,
        degradation_cfg: Optional[Dict] = None,
        eval_degradation: Optional[str] = None,
        eval_severity: Optional[int] = None,
    ):
        self.ds = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.pair_transform = SegPairTransform(transform_cfg or {}, train=train)
        self.train = train
        self.train_degrader = StochasticDegrader(degradation_cfg or {})
        self.eval_degrader = DeterministicDegrader(eval_degradation, eval_severity)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, mask = self.ds[idx]  # PIL.Image, PIL.Image
        if self.train:
            img = self.train_degrader(img)
        else:
            img = self.eval_degrader(img)
        img_t, mask_t = self.pair_transform(img, mask)
        return img_t, mask_t, idx
