"""Classical-CV backend (permissive core, no extra dependencies).

**Multi-hypothesis** slide detection: several candidate quads are proposed by
independent methods (bright-Otsu, hole-filled Otsu, Canny edges, HSV-value) and the
best is chosen by a composite score that rewards a quad whose border lies on a
strong image edge and separates a distinct interior from its surround — not merely
the brightest blob. This fixes the dominant failure of single-threshold detection
on color-cast projector shots (grabbing the bright *interior* image and cutting the
darker title, or swallowing bright ceiling lights); see ``IMPROVEMENT_PLAN.md``.

Works offline with only OpenCV/NumPy — no GPL, no model weights.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..geometry import aspect_score, order_corners
from .base import Detector


def _largest_quad(contours, img_area, min_frac=0.10) -> Optional[np.ndarray]:
    best, best_area = None, 0.0
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(c)
        if area < img_area * min_frac:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
                best, best_area = approx, area
                break
    return order_corners(best) if best is not None else None


# --------------------------------------------------------------------------- #
# Candidate generators — each proposes at most one quad from a different cue.
# --------------------------------------------------------------------------- #
def _cand_otsu(bgr: np.ndarray, area: int) -> Optional[np.ndarray]:
    """Bright region by global Otsu (the original C2 method)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((9, 9), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _largest_quad(cnts, area)


def _cand_fill(bgr: np.ndarray, area: int) -> Optional[np.ndarray]:
    """Otsu with a large close — swallows an internally-dark title band so the
    whole slide stays one region instead of collapsing to the bright interior."""
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(9, int(0.05 * max(h, w)) | 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _largest_quad(cnts, area)


def _cand_canny(bgr: np.ndarray, area: int) -> Optional[np.ndarray]:
    """Structural: the screen boundary as a closed Canny contour — brightness-
    independent, recovers slides whose interior is darker than a bright sub-image."""
    h, w = bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    k = max(5, int(0.02 * max(h, w)) | 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _largest_quad(cnts, area)


def _cand_value(bgr: np.ndarray, area: int) -> Optional[np.ndarray]:
    """Screen vs. surround on the HSV value channel with a large close — tolerant
    to strong color casts that skew a plain grayscale threshold."""
    h, w = bgr.shape[:2]
    v = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2], (7, 7), 0)
    _, th = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(9, int(0.06 * max(h, w)) | 1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _largest_quad(cnts, area)


_GENERATORS = (_cand_otsu, _cand_fill, _cand_canny, _cand_value)


# --------------------------------------------------------------------------- #
# Composite scorer — picks the *right* rectangle, not the brightest blob.
# --------------------------------------------------------------------------- #
def _edge_points(a: np.ndarray, b: np.ndarray, n: int = 40) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] * (1 - t) + b[None, :] * t


def score_quad(quad: np.ndarray, gray: np.ndarray, gradmag: np.ndarray,
               g90: float) -> float:
    """Higher = more likely the true slide boundary. Blends: how well the four
    borders sit on strong image gradients (edge support); how different the strip
    just inside each border is from just outside (interior/surround contrast);
    how *quiet* the region a little further outside is (a quad that cuts across the
    slide has slide content — texture — just outside it, so a textured exterior is
    penalized); slide-like aspect; sane area; and a penalty for corners on the
    image edge. Weights tuned on the review log (see IMPROVEMENT_PLAN.md)."""
    h, w = gray.shape
    q = order_corners(np.asarray(quad, dtype=np.float32))
    asp = aspect_score(q)
    area = cv2.contourArea(q) / (h * w)
    touch = sum(1 for x, y in q if x <= 2 or y <= 2 or x >= w - 3 or y >= h - 3)

    cx, cy = q.mean(axis=0)
    off = 0.03 * max(h, w)
    ext_off = 0.06 * max(h, w)
    esup, contrast, ext_tex = [], [], []
    for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
        pts = _edge_points(a, b)
        xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
        yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
        esup.append(float(gradmag[yi, xi].mean()))
        mid = (a + b) / 2.0
        nrm = np.array([cx, cy]) - mid
        nrm = nrm / (np.linalg.norm(nrm) + 1e-6)
        ins, out = pts + nrm * off, pts - nrm * off
        ii = (np.clip(ins[:, 1].astype(int), 0, h - 1), np.clip(ins[:, 0].astype(int), 0, w - 1))
        oo = (np.clip(out[:, 1].astype(int), 0, h - 1), np.clip(out[:, 0].astype(int), 0, w - 1))
        contrast.append(abs(float(gray[ii].mean()) - float(gray[oo].mean())) / 255.0)
        ext = pts - nrm * ext_off
        ee = (np.clip(ext[:, 1].astype(int), 0, h - 1), np.clip(ext[:, 0].astype(int), 0, w - 1))
        ext_tex.append(float(gradmag[ee].mean()))

    edge_support = float(np.mean(esup)) / (g90 + 1e-6)
    border_contrast = float(np.mean(contrast))
    exterior_texture = float(np.mean(ext_tex)) / (g90 + 1e-6)
    area_term = 1.0 if 0.08 <= area <= 0.97 else 0.0
    return (0.40 * min(edge_support, 1.0)
            + 0.25 * min(border_contrast * 2.0, 1.0)
            + 0.12 * asp
            + 0.10 * area_term
            - 0.25 * (touch / 4.0)
            - 0.20 * min(exterior_texture, 1.0))


class ClassicDetector(Detector):
    name = "classic"
    license_note = "OpenCV (Apache-2.0) + NumPy (BSD). No GPL, no model weights."

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = bgr.shape[:2]
        area = h * w

        candidates = []
        seen = []
        for gen in _GENERATORS:
            q = gen(bgr, area)
            if q is None:
                continue
            qo = order_corners(q)
            # cheap dedupe: skip a candidate within ~1% diagonal of an earlier one
            diag = np.hypot(h, w)
            if any(np.linalg.norm(qo - s, axis=1).max() < 0.01 * diag for s in seen):
                continue
            seen.append(qo)
            candidates.append(qo)

        if candidates:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradmag = cv2.magnitude(gx, gy)
            g90 = float(np.percentile(gradmag, 90))
            return max(candidates, key=lambda q: score_quad(q, gray, gradmag, g90))

        # last-resort fallback: rotated bounding box of the largest bright blob
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) >= area * 0.10:
                return order_corners(cv2.boxPoints(cv2.minAreaRect(c)))
        return None
