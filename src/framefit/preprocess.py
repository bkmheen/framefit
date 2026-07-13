"""Tone/contrast preprocessing used ONLY to help detection.

The preprocessed image is used to obtain coordinates; the final crop/warp is always
done from the untouched original. A1 (gamma/shadow lift) was the benchmark winner
for dark-auditorium slide photos.
"""
from __future__ import annotations

import cv2
import numpy as np


def identity(img: np.ndarray) -> np.ndarray:
    return img


def gamma_lift(img: np.ndarray, gamma: float = 0.42) -> np.ndarray:
    """A1 — raise dark tones so a dim screen/header separates from a dark room."""
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def clahe(img: np.ndarray, clip: float = 3.0, tiles: int = 8) -> np.ndarray:
    """A2 — local contrast on the L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


PREPROCESSORS = {"none": identity, "gamma": gamma_lift, "clahe": clahe}
