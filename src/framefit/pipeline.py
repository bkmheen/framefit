"""High-level pipeline: load → detect (on a preprocessed downscale) → warp/crop
from the untouched original → optional bezel inset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from . import io
from .backends import Detector, get_backend
from .geometry import aspect_ratio, aspect_score, inset_quad, order_corners, warp_from_quad


@dataclass
class Result:
    """Outcome of processing one image."""

    ok: bool
    image: Optional[np.ndarray]        # rectified BGR crop (None if detection failed)
    quad: Optional[np.ndarray]         # detected corners in ORIGINAL image space
    backend: str
    aspect_ratio: float = 0.0
    aspect_score: float = 0.0


def process_image(
    image: np.ndarray,
    backend: Union[str, Detector] = "auto",
    inset: float = 0.0,
    detect_max: int = 1400,
) -> Result:
    """Detect the slide in a BGR image and return the rectified full-frame crop."""
    det = backend if isinstance(backend, Detector) else get_backend(backend)

    h, w = image.shape[:2]
    scale = detect_max / max(h, w) if max(h, w) > detect_max else 1.0
    small = (
        cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )

    quad_small = det.detect(det.preprocess(small))
    if quad_small is None:
        return Result(False, None, None, det.name)

    quad = order_corners(quad_small / scale)
    quad = inset_quad(quad, inset)
    warped = warp_from_quad(image, quad)
    return Result(True, warped, quad, det.name, aspect_ratio(quad), aspect_score(quad))


def process_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend: Union[str, Detector] = "auto",
    inset: float = 0.0,
    detect_max: int = 1400,
    quality: int = 95,
) -> Result:
    """Load `src`, process it, and write the rectified crop to `dst`."""
    image = io.load_bgr(src)
    result = process_image(image, backend=backend, inset=inset, detect_max=detect_max)
    if result.ok and result.image is not None:
        io.save_bgr(result.image, dst, quality=quality)
    return result
