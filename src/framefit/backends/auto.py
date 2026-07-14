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
        #: name of the sub-backend that produced the most recent result. When this
        #: is not "docaligner", the deep-learning detector failed and we fell back
        #: to the classical core — a reliable low-confidence signal.
        self.last_source: Optional[str] = None
        #: composite detection score of the winning sub-detector (classical core
        #: only; None when DocAligner won or nothing was found). Calibration signal.
        self.last_score: Optional[float] = None
        self._subs: list[Detector] = []
        try:
            from .docaligner import DocAlignerDetector

            self._subs.append(DocAlignerDetector())
        except ImportError:
            pass
        self._subs.append(ClassicDetector())

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        best_quad, best_score, best_src, best_sub = None, -1.0, None, None
        for sub in self._subs:
            quad = sub.detect(sub.preprocess(bgr))
            if quad is None:
                continue
            score = aspect_score(quad)
            if score >= self.min_score:
                self.last_source = sub.name  # good enough; keep this backend
                self.last_score = getattr(sub, "last_score", None)
                return quad
            if score > best_score:
                best_quad, best_score, best_src, best_sub = quad, score, sub.name, sub
        self.last_source = best_src
        self.last_score = getattr(best_sub, "last_score", None)
        return best_quad
