"""Headless re-processing (no browser): regenerate crops from the logged
decisions. Because Module C (:func:`pipeline.process_manual`) and the feedback log
are UI-free, the entire reviewed dataset can be rebuilt offline — e.g. after a
warp-algorithm change, or to materialize crops on a machine that only synced the
JSONL log.

Reads every host shard under the shared dataset, and for each decision that has a
``final_quad`` re-warps the original (preferring the dataset's content-addressed
copy, falling back to the recorded ``source_path``) and writes the crop.

Usage:
    python -m framefit.batch_replay <output_dir> [--only-modified]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import feedback, io
from .pipeline import process_manual


def replay(outdir: str | Path, only_modified: bool = False) -> tuple[int, int]:
    """Rebuild crops from the log. Returns (written, skipped)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = feedback.review_root()
    written = skipped = 0

    for rec in feedback.read_log():
        final = rec.get("final_quad")
        if not final:                       # skip / no-crop decisions
            skipped += 1
            continue
        if only_modified and not rec.get("was_modified"):
            skipped += 1
            continue

        # Prefer the dataset's content-addressed original; fall back to source.
        src = None
        if rec.get("dataset_original"):
            cand = root / rec["dataset_original"]
            if cand.exists():
                src = cand
        if src is None and rec.get("source_path"):
            cand = Path(rec["source_path"])
            if cand.exists():
                src = cand
        if src is None:
            print(f"[miss] {rec.get('source_name')}: no original available",
                  file=sys.stderr)
            skipped += 1
            continue

        try:
            bgr = io.load_bgr(src)
            # final_quad is in the original's full-res px. The dataset copy may be
            # downscaled, so map the quad by the actual dimension ratio — this is
            # identity (ratio 1) for a true full-res source and the right scale for
            # a downscaled copy, with no need to know which one we loaded.
            fh, fw = bgr.shape[:2]
            sx = fw / rec["image_width"]
            sy = fh / rec["image_height"]
            quad = ([[x * sx, y * sy] for x, y in final]
                    if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3 else final)
            r = process_manual(bgr, quad, refine=False)
            dst = outdir / f"{Path(rec['source_name']).stem}_framefit.jpg"
            io.save_bgr(r.image, dst, quality=95)
            written += 1
        except Exception as e:  # noqa
            print(f"[error] {rec.get('source_name')}: {e}", file=sys.stderr)
            skipped += 1

    print(f"\nbatch_replay: {written} crops written to {outdir}, {skipped} skipped")
    return written, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="framefit.batch_replay",
                                 description="Regenerate crops from the logged "
                                             "review decisions (headless).")
    ap.add_argument("output_dir", help="where to write regenerated crops")
    ap.add_argument("--only-modified", action="store_true",
                    help="only replay decisions the human modified")
    args = ap.parse_args(argv)
    w, _ = replay(args.output_dir, only_modified=args.only_modified)
    return 0 if w else 1


if __name__ == "__main__":
    raise SystemExit(main())
