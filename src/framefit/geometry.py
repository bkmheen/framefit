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


def aspect_score_wh(w: float, h: float) -> float:
    """1.0 = matches a standard slide ratio; decreases with deviation."""
    if h <= 0 or w <= 0:
        return 0.0
    ar = w / h
    best = min(abs(ar - s) / s for s in STD_ASPECT_RATIOS)
    return max(0.0, 1.0 - best)


def aspect_score(quad: np.ndarray) -> float:
    """1.0 = matches a standard slide ratio; decreases with deviation."""
    w, h = quad_size(quad)
    return aspect_score_wh(w, h)


def trim_dark_margins(
    image: np.ndarray,
    dark_ratio: float = 0.12,
    var_max: float = 0.11,
    max_trim: float = 0.30,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Trim near-black border bands (bezel/room gap) from a rectified image.

    A border row/column is trimmed only while it is both near-black (mean below
    ``dark_ratio``) and near-uniform (std below ``var_max``), so content is never
    cut. ``dark_ratio`` is deliberately low (~0.12): the empty room/bezel above a
    slide sits near black (~0.09), while a dark slide header (e.g. a navy title bar,
    ~0.16+) is above it and must be preserved — a higher threshold would eat the
    header. Each edge is trimmed independently, capped at ``max_trim`` of the side
    length. Returns the cropped image and (top, bottom, left, right) pixels removed.
    Values are fractions of 255.
    """
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = g.shape
    dark, var = float(dark_ratio), float(var_max)

    def _scan(mean_prof: np.ndarray, std_prof: np.ndarray, limit: int) -> int:
        c = 0
        while c < limit and mean_prof[c] < dark and std_prof[c] < var:
            c += 1
        return c

    row_m, row_s = g.mean(1), g.std(1)
    col_m, col_s = g.mean(0), g.std(0)
    hcap, wcap = int(h * max_trim), int(w * max_trim)
    t = _scan(row_m, row_s, hcap)
    b = _scan(row_m[::-1], row_s[::-1], hcap)
    l = _scan(col_m, col_s, wcap)
    r = _scan(col_m[::-1], col_s[::-1], wcap)
    if t + b >= h or l + r >= w:  # safety: never trim everything
        return image, (0, 0, 0, 0)
    return image[t:h - b, l:w - r], (t, b, l, r)


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
