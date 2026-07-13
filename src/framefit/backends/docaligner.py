"""Deep-learning backend (DocAligner). Requires the `dl` extra.

Pairs the DocAligner heatmap corner model with an A1 gamma/shadow-lift preprocess,
which was the benchmark winner: lifting shadows reveals the whole projector screen
so the model locks onto the true corners. Model weights are downloaded at runtime
by DocAligner and are not redistributed by framefit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..geometry import order_corners
from ..preprocess import gamma_lift
from .base import Detector


class DocAlignerDetector(Detector):
    name = "docaligner"
    license_note = (
        "DocAligner code Apache-2.0; model weights downloaded at runtime under "
        "the DocAligner project's terms (not redistributed by framefit)."
    )

    def __init__(self):
        try:
            from docaligner import DocAligner
        except ImportError as e:
            raise ImportError(
                "The docaligner backend requires the 'dl' extra. "
                "Install it with:  pip install \"framefit[dl]\""
            ) from e
        self._model = DocAligner()

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        return gamma_lift(bgr)

    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        out = self._model(bgr)
        if out is None:
            return None
        poly = getattr(out, "doc_polygon", getattr(out, "polygon", out))
        poly = np.asarray(poly, dtype=np.float32)
        if poly.size == 0:
            return None
        poly = poly.reshape(-1, 2)
        if poly.shape[0] != 4:
            return None
        return order_corners(poly)
