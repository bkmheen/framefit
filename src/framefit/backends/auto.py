"""Auto backend: best-of over the available detectors.

Prefers the deep-learning DocAligner backend (when installed) and falls back to the
classical-CV core, picking whichever proposes the more slide-like quadrilateral
(by aspect-ratio score). Instantiated once and reused across a batch, so the DL
model is loaded a single time.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..geometry import aspect_score
from .base import Detector
from .classic import ClassicDetector


class AutoDetector(Detector):
    name = "auto"
    license_note = "picks docaligner if installed (its terms apply), else classic."

    def __init__(self, min_score: float = 0.80):
        self.min_score = min_score
        self._subs: list[Detector] = []
        try:
            from .docaligner import DocAlignerDetector

            self._subs.append(DocAlignerDetector())
        except ImportError:
            pass
        self._subs.append(ClassicDetector())

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        best_quad, best_score = None, -1.0
        for sub in self._subs:
            quad = sub.detect(sub.preprocess(bgr))
            if quad is None:
                continue
            score = aspect_score(quad)
            if score >= self.min_score:
                return quad  # good enough; keep the higher-priority backend's result
            if score > best_score:
                best_quad, best_score = quad, score
        return best_quad
