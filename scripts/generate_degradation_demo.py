from pathlib import Path
import sys
import json
import random
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_ROOT = PROJECT_ROOT / "data" / "VOCdevkit" / "VOC2012" / "JPEGImages"
OUT_DIR = PROJECT_ROOT / "artifacts" / "degradation_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


deg_module = None
project_degrade_callable = None
try:
    import src.degradations as deg_module
except Exception:
    pass

if deg_module is not None:
    for name in [
        "apply_degradation", "apply_degradations", "degrade_image",
        "degrade", "random_degradation", "build_degradation_pipeline"
    ]:
        if hasattr(deg_module, name) and callable(getattr(deg_module, name)):
            project_degrade_callable = getattr(deg_module, name)
            break

project_apply_degradation = None
if deg_module is not None and hasattr(deg_module, "apply_degradation"):
    project_apply_degradation = getattr(deg_module, "apply_degradation")


def to_uint8_np(img):
    arr = np.array(img)
    return np.clip(arr, 0, 255).astype(np.uint8)


def from_np(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


GAUSSIAN_STD = {1: 5.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 45.0}
MOTION_BLUR_KS = {1: 3, 2: 5, 3: 9, 4: 13, 5: 17}
JPEG_QUALITY = {1: 85, 2: 65, 3: 45, 4: 30, 5: 15}
SEVERITY_LEVELS = [1, 3, 5]
DEG_SEVERITY_TYPES = ["gaussian_noise", "motion_blur", "jpeg"]


def add_gaussian_noise(img, severity=3):
    arr = to_uint8_np(img).astype(np.float32)
    sigma = GAUSSIAN_STD.get(int(severity), 20.0)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    return from_np(arr + noise)


def add_motion_blur(img, severity=3):
    ksize = MOTION_BLUR_KS.get(int(severity), 11)
    if cv2 is None:
        return img.filter(ImageFilter.GaussianBlur(radius=max(1, ksize // 6)))
    arr = to_uint8_np(img)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    return from_np(cv2.filter2D(arr, -1, kernel))


def add_defocus_blur(img, radius=2):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_jpeg_compression(img, severity=3):
    import io

    quality = JPEG_QUALITY.get(int(severity), 45)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def reduce_resolution(img, scale=0.5):
    w, h = img.size
    img2 = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return img2.resize((w, h), Image.BILINEAR)


def adjust_low_light(img, factor=0.45):
    arr = to_uint8_np(img).astype(np.float32)
    return from_np(arr * factor)


def fallback_degradation(img, dtype, severity=3):
    dtype = str(dtype)
    if dtype == "gaussian_noise":
        return add_gaussian_noise(img, severity)
    if dtype == "motion_blur":
        return add_motion_blur(img, severity)
    if dtype == "jpeg":
        return add_jpeg_compression(img, severity)
    raise ValueError(f"Unsupported fallback degradation: {dtype}")


FALLBACK_EXTRA = {
    "defocus_blur": lambda img, severity: add_defocus_blur(img, radius=2),
    "downsample_x0.5": lambda img, severity: reduce_resolution(img, scale=0.5),
    "low_light": lambda img, severity: adjust_low_light(img, factor=0.45),
}


def fallback_degradations(img):
    return {
        "gaussian_noise": add_gaussian_noise(img),
        "motion_blur": add_motion_blur(img),
        "defocus_blur": add_defocus_blur(img),
        "jpeg": add_jpeg_compression(img),
        "downsample_x0.5": reduce_resolution(img),
        "low_light": adjust_low_light(img),
    }


def normalize_project_output(out):
    if out is None:
        return None
    if isinstance(out, dict):
        rst = {}
        for k, v in out.items():
            if isinstance(v, Image.Image):
                rst[str(k)] = v.convert("RGB")
            elif isinstance(v, np.ndarray):
                rst[str(k)] = from_np(v)
        return rst if rst else None
    if isinstance(out, Image.Image):
        return {"project_degraded": out.convert("RGB")}
    if isinstance(out, np.ndarray):
        return {"project_degraded": from_np(out)}
    return None


def run_project_degradation_if_possible(img):
    if project_degrade_callable is None:
        return None
    tries = [
        lambda: project_degrade_callable(img),
        lambda: project_degrade_callable(np.array(img)),
        lambda: project_degrade_callable(image=img),
        lambda: project_degrade_callable(image=np.array(img)),
    ]
    for fn in tries:
        try:
            return fn()
        except Exception:
            pass
    return None


def apply_project_severity(img, dtype, severity):
    if project_apply_degradation is None:
        return None
    try:
        return project_apply_degradation(img, dtype, severity)
    except Exception:
        return None


def degrade_with_severity(img, dtype, severity):
    project_out = apply_project_severity(img, dtype, severity)
    if project_out is not None:
        return project_out.convert("RGB")
    if dtype in FALLBACK_EXTRA:
        return FALLBACK_EXTRA[dtype](img, severity)
    return fallback_degradation(img, dtype, severity)


def make_grid(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    po = normalize_project_output(run_project_degradation_if_possible(img))
    if po:
        degs = po
        source = "project"
    else:
        degs = fallback_degradations(img)
        source = "fallback"

    items = [("original", img)] + list(degs.items())
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (name, im) in enumerate(items, 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{img_path.name} | source={source}")
    fig.tight_layout()

    out = OUT_DIR / f"{img_path.stem}_degradation_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "image": img_path.name,
        "output": str(out),
        "source": source,
        "degradations": list(degs.keys()),
    }


def make_severity_grid(img_path: Path, dtype: str, severities=SEVERITY_LEVELS):
    img = Image.open(img_path).convert("RGB")
    items = [(f"severity={sev}", degrade_with_severity(img, dtype, sev)) for sev in severities]
    cols = len(items)
    fig = plt.figure(figsize=(4 * cols, 4))
    for i, (label, im) in enumerate(items, 1):
        ax = fig.add_subplot(1, cols, i)
        ax.imshow(im)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(f"{img_path.name} // {dtype} severity")
    fig.tight_layout()

    out = OUT_DIR / f"{img_path.stem}_{dtype}_severity_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def make_severity_grids(img_path: Path):
    return [str(make_severity_grid(img_path, dtype)) for dtype in DEG_SEVERITY_TYPES]


def main():
    imgs = sorted(DATA_ROOT.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No jpg in {DATA_ROOT}")
    random.seed(271)
    sample_paths = random.sample(imgs, k=min(3, len(imgs)))
    summary = []
    for p in sample_paths:
        grid = make_grid(p)
        severity_paths = make_severity_grids(p)
        grid["severity_grids"] = severity_paths
        summary.append(grid)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
from pathlib import Path
import sys
import json
import random
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_ROOT = PROJECT_ROOT / "data" / "VOCdevkit" / "VOC2012" / "JPEGImages"
OUT_DIR = PROJECT_ROOT / "artifacts" / "degradation_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

deg_module = None
project_degrade_callable = None
try:
    import src.degradations as deg_module
except Exception:
    pass

if deg_module is not None:
    for name in [
        "apply_degradation", "apply_degradations", "degrade_image",
        "degrade", "random_degradation", "build_degradation_pipeline"
    ]:
        if hasattr(deg_module, name) and callable(getattr(deg_module, name)):
            project_degrade_callable = getattr(deg_module, name)
            break


def to_uint8_np(img):
    arr = np.array(img)
    return np.clip(arr, 0, 255).astype(np.uint8)


def from_np(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_gaussian_noise(img, sigma=18):
    arr = to_uint8_np(img).astype(np.float32)
    return from_np(arr + np.random.normal(0, sigma, arr.shape))


def add_motion_blur(img, ksize=13):
    if cv2 is None:
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    arr = to_uint8_np(img)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    return from_np(cv2.filter2D(arr, -1, kernel))


def add_defocus_blur(img, radius=2):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_jpeg_compression(img, quality=25):
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def reduce_resolution(img, scale=0.5):
    w, h = img.size
    img2 = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return img2.resize((w, h), Image.BILINEAR)


def adjust_low_light(img, factor=0.45):
    arr = to_uint8_np(img).astype(np.float32)
    return from_np(arr * factor)


def fallback_degradations(img):
    return {
        "gaussian_noise": add_gaussian_noise(img),
        "motion_blur": add_motion_blur(img),
        "defocus_blur": add_defocus_blur(img),
        "jpeg_25": add_jpeg_compression(img),
        "downsample_x0.5": reduce_resolution(img),
        "low_light": adjust_low_light(img),
    }


def normalize_project_output(out):
    if out is None:
        return None
    if isinstance(out, dict):
        rst = {}
        for k, v in out.items():
            if isinstance(v, Image.Image):
                rst[str(k)] = v.convert("RGB")
            elif isinstance(v, np.ndarray):
                rst[str(k)] = from_np(v)
        return rst if rst else None
    if isinstance(out, Image.Image):
        return {"project_degraded": out.convert("RGB")}
    if isinstance(out, np.ndarray):
        return {"project_degraded": from_np(out)}
    return None


def run_project_degradation_if_possible(img):
    if project_degrade_callable is None:
        return None
    tries = [
        lambda: project_degrade_callable(img),
        lambda: project_degrade_callable(np.array(img)),
        lambda: project_degrade_callable(image=img),
        lambda: project_degrade_callable(image=np.array(img)),
    ]
    for fn in tries:
        try:
            return fn()
        except Exception:
            pass
    return None


def make_grid(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    po = normalize_project_output(run_project_degradation_if_possible(img))
    if po:
        degs = po
        source = "project"
    else:
        degs = fallback_degradations(img)
        source = "fallback"

    items = [("original", img)] + list(degs.items())
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (name, im) in enumerate(items, 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{img_path.name} | source={source}")
    fig.tight_layout()

    out = OUT_DIR / f"{img_path.stem}_degradation_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "image": img_path.name,
        "output": str(out),
        "source": source,
        "degradations": list(degs.keys()),
    }


def main():
    imgs = sorted(DATA_ROOT.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No jpg in {DATA_ROOT}")
    random.seed(271)
    sample_paths = random.sample(imgs, k=min(3, len(imgs)))
    summary = [make_grid(p) for p in sample_paths]
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
