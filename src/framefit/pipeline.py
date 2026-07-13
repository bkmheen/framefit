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
from .geometry import (
    aspect_score_wh,
    inset_quad,
    order_corners,
    trim_dark_margins,
    warp_from_quad,
)


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
    expand: float = 0.04,
    detect_max: int = 1400,
    refine: bool = True,
) -> Result:
    """Detect the slide in a BGR image and return the rectified full-frame crop.

    ``expand`` grows the detected quad outward by this fraction before warping — a
    safety margin so a slightly-inaccurate detection never crops into content (e.g.
    a title flush to the slide's top edge). The following ``refine`` pass reclaims
    the added margin wherever it is genuinely empty (dark), so on the common case
    the result stays tight while content is protected.

    When ``refine`` is set (default), uniformly-dark border bands left by an
    imprecise edge (typically the top, above a dark header) are trimmed off the
    rectified image so the slide fills the frame exactly.
    """
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
    quad = inset_quad(quad, inset - expand)  # net: shrink by inset, grow by expand
    warped = warp_from_quad(image, quad)
    if refine:
        warped, _ = trim_dark_margins(warped)

    oh, ow = warped.shape[:2]
    return Result(True, warped, quad, det.name, ow / oh, aspect_score_wh(ow, oh))


def process_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    backend: Union[str, Detector] = "auto",
    inset: float = 0.0,
    expand: float = 0.04,
    detect_max: int = 1400,
    refine: bool = True,
    quality: int = 95,
) -> Result:
    """Load `src`, process it, and write the rectified crop to `dst`."""
    image = io.load_bgr(src)
    result = process_image(image, backend=backend, inset=inset, expand=expand,
                           detect_max=detect_max, refine=refine)
    if result.ok and result.image is not None:
        io.save_bgr(result.image, dst, quality=quality)
    return result
