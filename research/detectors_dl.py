"""Deep-learning slide/document corner detector (C5) for framefit research.

Wraps DocAligner (DocsaidLab) heatmap-regression corner model. Kept separate from
the classical detectors so the core benchmark stays dependency-light and offline.

The model + ONNX weights are downloaded on first use (~cached in the package dir).
"""
from __future__ import annotations

import numpy as np

from detectors import order_corners

_model = None


def _get_model():
    global _model
    if _model is None:
        from docaligner import DocAligner  # heavy import; lazy
        _model = DocAligner()
    return _model


def detect_docaligner(img: np.ndarray):
    """Return ordered 4x2 corners (TL,TR,BR,BL) in input-image space, or None."""
    m = _get_model()
    out = m(img)

    # Normalize the various possible return shapes to a (4,2) array.
    poly = None
    if out is None:
        return None
    if hasattr(out, "doc_polygon"):
        poly = np.asarray(out.doc_polygon, dtype=np.float32)
    elif hasattr(out, "polygon"):
        poly = np.asarray(out.polygon, dtype=np.float32)
    else:
        poly = np.asarray(out, dtype=np.float32)

    if poly is None or poly.size == 0:
        return None
    poly = poly.reshape(-1, 2)
    if poly.shape[0] != 4:
        return None
    # DocAligner returns points in image pixel coords already.
    return order_corners(poly)


DL_DETECTORS = {"C5_docaligner": detect_docaligner}
