from __future__ import annotations
import io
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

GAUSSIAN_STD = {1: 5.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 45.0}  # 0-255 scale
MOTION_BLUR_K = {1: 3, 2: 5, 3: 9, 4: 13, 5: 17}
JPEG_QUALITY = {1: 85, 2: 65, 3: 45, 4: 30, 5: 15}

def _to_np_uint8(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)

def _to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    arr = _to_np_uint8(img).astype(np.float32)
    std = GAUSSIAN_STD[int(severity)]
    noise = np.random.normal(0, std, size=arr.shape).astype(np.float32)
    return _to_pil(arr + noise)

def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    if k.sum() <= 0:
        k[length // 2, :] = 1.0
    k = k / (k.sum() + 1e-8)
    return k

def apply_motion_blur(img: Image.Image, severity: int, angle_deg: Optional[float] = None) -> Image.Image:
    arr = _to_np_uint8(img)
    ksize = MOTION_BLUR_K[int(severity)]
    angle_deg = random.uniform(0, 180) if angle_deg is None else angle_deg
    kernel = _motion_kernel(ksize, angle_deg)
    out = cv2.filter2D(arr, -1, kernel)
    return _to_pil(out)

def apply_jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    q = JPEG_QUALITY[int(severity)]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out

def apply_degradation(img: Image.Image, dtype: str, severity: int) -> Image.Image:
    if dtype == "gaussian_noise":
        return apply_gaussian_noise(img, severity)
    if dtype == "motion_blur":
        return apply_motion_blur(img, severity)
    if dtype == "jpeg":
        return apply_jpeg_compression(img, severity)
    raise ValueError(f"Unsupported degradation type: {dtype}")

@dataclass
class DegradationConfig:
    enable: bool = False
    prob: float = 0.0
    types: Tuple[str, ...] = ("gaussian_noise", "motion_blur", "jpeg")
    min_severity: int = 1
    max_severity: int = 5
    one_of: bool = True
    max_compose: int = 2
    seed: Optional[int] = None

class StochasticDegrader:
    def __init__(self, cfg: Dict):
        self.cfg = DegradationConfig(
            enable=bool(cfg.get("enable", False)),
            prob=float(cfg.get("prob", 0.0)),
            types=tuple(cfg.get("types", ["gaussian_noise", "motion_blur", "jpeg"])),
            min_severity=int(cfg.get("min_severity", 1)),
            max_severity=int(cfg.get("max_severity", 5)),
            one_of=bool(cfg.get("one_of", True)),
            max_compose=int(cfg.get("max_compose", 2)),
            seed=cfg.get("seed", None),
        )
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

    def _sample_severity(self) -> int:
        return random.randint(self.cfg.min_severity, self.cfg.max_severity)

    def __call__(self, img: Image.Image) -> Image.Image:
        if (not self.cfg.enable) or (random.random() > self.cfg.prob):
            return img
        if self.cfg.one_of:
            dtype = random.choice(self.cfg.types)
            return apply_degradation(img, dtype, self._sample_severity())
        n = random.randint(1, min(len(self.cfg.types), self.cfg.max_compose))
        dtypes = random.sample(list(self.cfg.types), n)
        out = img
        for d in dtypes:
            out = apply_degradation(out, d, self._sample_severity())
        return out

class DeterministicDegrader:
    """用于评估阶段：指定退化类型与强度，保证可复现。"""
    def __init__(self, dtype: Optional[str] = None, severity: Optional[int] = None):
        self.dtype = dtype
        self.severity = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.dtype is None:
            return img
        if self.severity is None:
            raise ValueError("severity must be set when dtype is set")
        return apply_degradation(img, self.dtype, int(self.severity))
