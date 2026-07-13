"""Classical-CV backend (permissive core, no extra dependencies).

Brightness-threshold quad detection (benchmark C2) with a rotated-bounding-box
fallback (C4). Works offline with only OpenCV/NumPy — no GPL, no model weights.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..geometry import order_corners
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


class ClassicDetector(Detector):
    name = "classic"
    license_note = "OpenCV (Apache-2.0) + NumPy (BSD). No GPL, no model weights."

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = bgr.shape[:2]
        area = h * w
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = np.ones((9, 9), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quad = _largest_quad(cnts, area)
        if quad is not None:
            return quad
        # C4 fallback: rotated bounding box of the largest bright blob
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) >= area * 0.10:
                return order_corners(cv2.boxPoints(cv2.minAreaRect(c)))
        return None
