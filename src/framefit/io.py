"""Image loading/saving. HEIC/HEIF support is optional (the `heic` extra)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HEIC_SUFFIXES = {".heic", ".heif", ".hif"}
_heif_registered = False


def _ensure_heif():
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif  # noqa
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "HEIC/HEIF input requires the 'heic' extra. "
            "Install it with:  pip install \"framefit[heic]\""
        ) from e
    pillow_heif.register_heif_opener()
    _heif_registered = True


def load_bgr(path: str | Path) -> np.ndarray:
    """Load an image as an OpenCV BGR uint8 array."""
    path = Path(path)
    if path.suffix.lower() in HEIC_SUFFIXES:
        _ensure_heif()
        from PIL import Image

        rgb = np.array(Image.open(path).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # fallback via Pillow for formats OpenCV can't read
    from PIL import Image

    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_bgr(image: np.ndarray, path: str | Path, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise IOError(f"Failed to encode image to {path}")
    path.write_bytes(buf.tobytes())
