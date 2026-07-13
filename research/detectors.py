"""Candidate slide-detection strategies for framefit research.

Each detector takes a BGR image (numpy, uint8) and returns either a 4x2 float32
array of corner points (ordered TL, TR, BR, BL) in the coordinate space of the
*input image*, or None if it fails to find a plausible quad.

These are classical-CV baselines used to benchmark detection quality on the
Optica slide photos before we commit to a final approach.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(quad.astype(np.float32)))


def _largest_quad_from_contours(contours, img_area, min_frac=0.10):
    """Return the largest 4-point convex quad among contours, or None."""
    best = None
    best_area = 0.0
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(c)
        if area < img_area * min_frac:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if area > best_area:
                    best_area = area
                    best = approx
                break
    if best is None:
        return None
    return order_corners(best)


# ---------------------------------------------------------------------------
# C1 — Canny + largest quad contour (classic document scanner)
# ---------------------------------------------------------------------------
def detect_canny(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _largest_quad_from_contours(cnts, h * w)


# ---------------------------------------------------------------------------
# C2 — Brightness threshold (Otsu) + largest bright quad  [scene-tailored]
# ---------------------------------------------------------------------------
def detect_bright(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((9, 9), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = _largest_quad_from_contours(cnts, h * w)
    if quad is not None:
        return quad
    # fallback: convex hull of the largest bright blob, then approx
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > h * w * 0.10:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.04, 0.06, 0.1):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    return order_corners(approx)
    return None


# ---------------------------------------------------------------------------
# C3 — Hough line 4-edge intersection
# ---------------------------------------------------------------------------
def _line_intersection(l1, l2):
    (x1, y1, x2, y2) = l1
    (x3, y3, x4, y4) = l2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return (px, py)


def detect_hough(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min(h, w) // 3, maxLineGap=40)
    if lines is None:
        return None
    lines = np.asarray(lines).reshape(-1, 4)  # cv2 5.0 may return (N,4) or (N,1,4)
    horis, verts = [], []
    for l in lines:
        x1, y1, x2, y2 = l
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) < 30 or abs(ang) > 150:
            horis.append(l)
        elif 60 < abs(ang) < 120:
            verts.append(l)
    if len(horis) < 2 or len(verts) < 2:
        return None
    horis.sort(key=lambda l: (l[1] + l[3]) / 2)
    verts.sort(key=lambda l: (l[0] + l[2]) / 2)
    top, bot = horis[0], horis[-1]
    left, right = verts[0], verts[-1]
    corners = []
    for hl in (top, bot):
        for vl in (left, right):
            p = _line_intersection(hl, vl)
            if p is None:
                return None
            corners.append(p)
    quad = np.array(corners, dtype=np.float32)
    if quad_area(order_corners(quad)) < h * w * 0.10:
        return None
    return order_corners(quad)


# ---------------------------------------------------------------------------
# C4 — minAreaRect on largest bright blob (rotation-only baseline)
# ---------------------------------------------------------------------------
def detect_minarearect(img: np.ndarray):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), 2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < h * w * 0.10:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return order_corners(box)


DETECTORS = {
    "C1_canny": detect_canny,
    "C2_bright": detect_bright,
    "C3_hough": detect_hough,
    "C4_minarearect": detect_minarearect,
}
