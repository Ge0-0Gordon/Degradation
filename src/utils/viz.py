from __future__ import annotations
import numpy as np
from PIL import Image

VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

def mask_to_color(mask: np.ndarray, ignore_index: int = 255) -> Image.Image:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(VOC_COLORMAP))
    out[valid] = VOC_COLORMAP[mask[valid]]
    out[mask == ignore_index] = [255, 255, 255]
    return Image.fromarray(out)

def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    mask_rgb = np.array(mask_to_color(mask), dtype=np.float32)
    out = img_np * (1 - alpha) + mask_rgb * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
