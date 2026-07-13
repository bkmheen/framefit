"""Backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..preprocess import identity


class Detector(ABC):
    """Base class for slide/document corner detectors.

    The pipeline downscales the image for speed, calls ``preprocess`` (a tone
    transform used only to aid detection), then ``detect``. Coordinates are mapped
    back to full resolution by the pipeline; the warp always uses the original.
    """

    #: license category of this backend, surfaced to users
    name: str = "base"
    license_note: str = ""

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Tone transform applied to the detection image (default: none)."""
        return identity(bgr)

    @abstractmethod
    def detect(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return ordered 4x2 corners (TL,TR,BR,BL) in `bgr` space, or None."""
        raise NotImplementedError
