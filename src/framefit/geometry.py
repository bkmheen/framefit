"""Geometry helpers: corner ordering, perspective warp, aspect scoring."""
from __future__ import annotations

import cv2
import numpy as np

# Standard slide/document aspect ratios we score detections against.
STD_ASPECT_RATIOS = (16 / 9, 16 / 10, 4 / 3, 3 / 2)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(quad, dtype=np.float32)))


def quad_size(quad: np.ndarray) -> tuple[int, int]:
    """Target (width, height) for the rectified output of a quad."""
    tl, tr, br, bl = np.asarray(quad, dtype=np.float32)
    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    return max(int(round(w)), 1), max(int(round(h)), 1)


def aspect_ratio(quad: np.ndarray) -> float:
    w, h = quad_size(quad)
    return w / h if h else 0.0


def aspect_score(quad: np.ndarray) -> float:
    """1.0 = matches a standard slide ratio; decreases with deviation."""
    ar = aspect_ratio(quad)
    if ar <= 0:
        return 0.0
    best = min(abs(ar - s) / s for s in STD_ASPECT_RATIOS)
    return max(0.0, 1.0 - best)


def inset_quad(quad: np.ndarray, frac: float) -> np.ndarray:
    """Move each corner toward the centroid by `frac` (e.g. 0.01 trims a bezel)."""
    if frac <= 0:
        return np.asarray(quad, dtype=np.float32)
    quad = np.asarray(quad, dtype=np.float32)
    c = quad.mean(axis=0)
    return (quad + (c - quad) * frac).astype(np.float32)


def warp_from_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-rectify the region bounded by `quad` into an upright rectangle."""
    quad = order_corners(quad)
    w, h = quad_size(quad)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, m, (w, h))
