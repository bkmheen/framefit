"""framefit command-line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .backends import get_backend
from .pipeline import process_file

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
                ".webp", ".heic", ".heif", ".hif"}


def _expand_inputs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out += sorted(f for f in path.iterdir()
                          if f.suffix.lower() in IMG_SUFFIXES)
        else:
            out.append(path)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="framefit",
        description="Detect a slide/document in a photo, correct perspective, "
                    "and crop to a clean full-frame image.",
    )
    p.add_argument("inputs", nargs="+", help="image file(s) or directory(ies)")
    p.add_argument("-o", "--output", default="framefit_out",
                   help="output directory (default: framefit_out)")
    p.add_argument("-b", "--backend", default="auto",
                   choices=["auto", "classic", "docaligner"],
                   help="detection backend (default: auto)")
    p.add_argument("--inset", type=float, default=0.0,
                   help="trim a bezel by moving corners inward, fraction e.g. 0.01")
    p.add_argument("-f", "--format", default="jpg",
                   choices=["jpg", "png"], help="output format (default: jpg)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    p.add_argument("--detect-max", type=int, default=1400,
                   help="longest side used for detection (default: 1400)")
    p.add_argument("--version", action="version", version=f"framefit {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = _expand_inputs(args.inputs)
    if not files:
        print("framefit: no input images found", file=sys.stderr)
        return 2

    backend = get_backend(args.backend)  # instantiate once (loads model if any)
    outdir = Path(args.output)
    ok_count = 0
    for f in files:
        dst = outdir / f"{f.stem}_framefit.{args.format}"
        try:
            r = process_file(f, dst, backend=backend, inset=args.inset,
                             detect_max=args.detect_max, quality=args.quality)
        except Exception as e:  # noqa
            print(f"[error] {f.name}: {e}", file=sys.stderr)
            continue
        if r.ok:
            ok_count += 1
            print(f"[ok]   {f.name} -> {dst}  (AR {r.aspect_ratio:.2f}, "
                  f"score {r.aspect_score:.2f}, {r.backend})")
        else:
            print(f"[miss] {f.name}: no slide detected", file=sys.stderr)

    print(f"\nframefit: {ok_count}/{len(files)} succeeded ({backend.name} backend)")
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
