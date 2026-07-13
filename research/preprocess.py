"""Tone/contrast preprocessing variants used to make the slide boundary easier
for the DL corner detector (C5) to lock onto.

Central idea: the preprocessed image is used ONLY to obtain coordinates. The final
crop/warp is always done from the untouched original. These transforms attack the
observed failure mode — the dark navy header blending into the dark auditorium.
"""
from __future__ import annotations

import cv2
import numpy as np


def identity(img: np.ndarray) -> np.ndarray:
    return img


# ---------------------------------------------------------------------------
# A1 — shadow / gamma lift: raise dark tones so the screen border & navy header
#      separate from the near-black room. Highlights are left mostly intact.
# ---------------------------------------------------------------------------
def a1_gamma_lift(img: np.ndarray, gamma: float = 0.42) -> np.ndarray:
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


# ---------------------------------------------------------------------------
# A2 — CLAHE local contrast on the L channel: crisps the screen-vs-room edge
#      without blowing out the bright content.
# ---------------------------------------------------------------------------
def a2_clahe(img: np.ndarray, clip: float = 3.0, tiles: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# A3 — screen-emission isolation: the projected screen emits light, the room does
#      not. Keep only the lit region (low threshold, captures the dim navy header
#      but not the black room), stretch its contrast, black out everything else.
#      Gives the detector a near-silhouette rectangle.
# ---------------------------------------------------------------------------
def a3_screen_isolate(img: np.ndarray, thresh: int = 28) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray_b, thresh, 255, cv2.THRESH_BINARY)
    # keep the largest connected lit region (the screen), drop stray reflections
    k = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [big], -1, 255, -1)
    out = cv2.bitwise_and(img, img, mask=mask)
    # contrast-stretch inside the lit region
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    m = mask > 0
    if m.any():
        lo, hi = np.percentile(l[m], (2, 98))
        if hi > lo:
            l = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            l[~m] = 0
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


PREPROCESSORS = {
    "raw": identity,
    "A1_gamma": a1_gamma_lift,
    "A2_clahe": a2_clahe,
    "A3_screeniso": a3_screen_isolate,
}
