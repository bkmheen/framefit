"""Detection backends.

A backend takes a BGR image and returns four ordered corners (TL,TR,BR,BL) of the
slide/document, or None. `get_backend` selects one by name; "auto" prefers the DL
backend when installed and falls back to the classical-CV core.
"""
from __future__ import annotations

from .base import Detector


def get_backend(name: str = "auto") -> Detector:
    name = (name or "auto").lower()
    if name in ("classic", "cv"):
        from .classic import ClassicDetector

        return ClassicDetector()
    if name in ("docaligner", "dl"):
        from .docaligner import DocAlignerDetector

        return DocAlignerDetector()
    if name == "auto":
        from .auto import AutoDetector

        return AutoDetector()
    raise ValueError(f"Unknown backend: {name!r} (use auto|classic|docaligner)")


__all__ = ["Detector", "get_backend"]
