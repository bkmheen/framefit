"""framefit — detect a slide/document in a photo, correct perspective, crop to full frame."""
from __future__ import annotations

__version__ = "0.5.0"

from .pipeline import Result, process_file, process_image  # noqa: E402
from .backends import get_backend  # noqa: E402

__all__ = ["__version__", "Result", "process_image", "process_file", "get_backend"]
