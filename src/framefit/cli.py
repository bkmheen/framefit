"""framefit command-line interface."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .backends import get_backend
from .pipeline import process_file

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
                ".webp", ".heic", ".heif", ".hif"}


def _expand_inputs(paths: list[str], exts: str | None = None,
                   recurse: bool = False, under: str | None = None) -> list[Path]:
    """Expand directory inputs to image files. ``exts`` (comma-separated, e.g.
    "heic" or "heic,jpg") restricts a directory scan to those extensions; when
    omitted every known image type is included. ``recurse`` walks every
    subdirectory. ``under`` keeps only files that live under a directory with
    that name (e.g. ``under="source"`` selects every HEIC below any ``source/``
    folder, ignoring images that sit elsewhere). Explicit file paths are kept
    as-is regardless of ``exts``/``under``."""
    allow = None
    if exts:
        allow = {"." + e.strip().lower().lstrip(".")
                 for e in exts.split(",") if e.strip()}
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            sel = allow or IMG_SUFFIXES
            it = path.rglob("*") if recurse else path.iterdir()
            for f in it:
                if not f.is_file() or f.suffix.lower() not in sel:
                    continue
                if under is not None and under not in f.parts:
                    continue
                out.append(f)
            out.sort()
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
    p.add_argument("-o", "--output", default=None,
                   help="output directory. Giving this OPTS OUT of the default "
                        "beside-the-source rule and writes <stem>_framefit.<fmt> "
                        "here instead (falls back to ./framefit_out for the "
                        "--pick/--review pages when omitted).")
    p.add_argument("-b", "--backend", default="auto",
                   choices=["auto", "classic", "docaligner"],
                   help="detection backend (default: auto)")
    p.add_argument("--inset", type=float, default=0.0,
                   help="trim a bezel by moving corners inward, fraction e.g. 0.01")
    p.add_argument("--expand", type=float, default=0.04,
                   help="grow the detected quad outward as a safety margin so a "
                        "slightly-off detection never crops into content (default: 0.04)")
    p.add_argument("--no-refine", dest="refine", action="store_false",
                   help="disable post-warp trimming of dark border margins")
    p.add_argument("--review-threshold", type=float, default=0.90,
                   help="flag results below this aspect-ratio score for review "
                        "in framefit_report.tsv (default: 0.90)")
    p.add_argument("--corners", default=None,
                   help="manual mode (single image): 4 'x,y' corners TL TR BR BL in "
                        "original-image pixels, e.g. --corners \"10,20 900,25 890,600 5,590\"")
    p.add_argument("--pick", action="store_true",
                   help="don't process — write an HTML corner-picker per input to "
                        "the output dir (use for images flagged for review)")
    p.add_argument("--review", action="store_true",
                   help="interactive review: open a local browser page to confirm "
                        "or edit the auto-detected corners per image, then crop. "
                        "Logs every decision to the shared learning dataset.")
    p.add_argument("--display-max", type=int, default=1400,
                   help="longest side shown in the review page (default: 1400)")
    p.add_argument("--beside", action="store_true",
                   help="write each result next to its source with the SAME name "
                        "and a .jpg extension (e.g. src/IMG.HEIC -> src/IMG.jpg). "
                        "This is the DEFAULT when no -o/--output is given; the flag "
                        "is kept for explicitness and back-compat.")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output file (otherwise it is skipped)")
    p.add_argument("--ext", default=None,
                   help="when an input is a directory, only process files with "
                        "this extension (comma-separated ok, e.g. --ext heic or "
                        "--ext heic,jpg). Default: every known image type.")
    p.add_argument("--only-flagged", action="store_true",
                   help="review mode: only stop on low-confidence/flagged images; "
                        "confident detections are auto-accepted and cropped without "
                        "prompting")
    p.add_argument("--recurse", action="store_true",
                   help="recurse into subdirectories when an input is a directory "
                        "(e.g. process every HEIC under any source/ folder)")
    p.add_argument("--under", default=None, metavar="DIRNAME",
                   help="with --recurse, only include files that live under a "
                        "directory named DIRNAME (e.g. --under source)")
    p.add_argument("--skip-decided", action="store_true",
                   help="skip images already decided in the review log — protects "
                        "hand-corrected crops from being overwritten on a re-run")
    p.add_argument("-f", "--format", default="jpg",
                   choices=["jpg", "png"], help="output format (default: jpg)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    p.add_argument("--detect-max", type=int, default=1400,
                   help="longest side used for detection (default: 1400)")
    p.add_argument("--version", action="version", version=f"framefit {__version__}")
    return p


def _parse_corners(s: str):
    pts = [tuple(float(v) for v in p.split(",")) for p in s.split()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise ValueError("--corners needs exactly 4 'x,y' points")
    return pts


def _dst_for(src: Path, outdir: Path, beside: bool, fmt: str) -> Path:
    """Where the crop for ``src`` is written. ``--beside`` puts it next to the
    source with the same stem and a .jpg extension; otherwise under ``outdir`` as
    ``<stem>_framefit.<fmt>``."""
    if beside:
        return src.parent / f"{src.stem}.jpg"
    return outdir / f"{src.stem}_framefit.{fmt}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = _expand_inputs(args.inputs, args.ext, recurse=args.recurse,
                           under=args.under)
    if not files:
        print("framefit: no input images found", file=sys.stderr)
        return 2

    # Beside-the-source (same filename, .jpg) is the default output rule so a
    # bare `framefit <imgs>` — the shape a human or another agent uses — never
    # invents an arbitrary output directory. Passing -o/--output opts out.
    beside = args.beside or args.output is None
    outdir = Path(args.output) if args.output is not None else Path("framefit_out")

    # --pick: write an HTML corner-picker per input, don't process
    if args.pick:
        from .picker import write_picker_html
        for f in files:
            html = write_picker_html(f, outdir / f"{f.stem}_pick.html", out_dir=str(outdir))
            print(f"[pick] {f.name} -> {html}")
        print(f"\nOpen the *_pick.html file(s), click 4 corners, run the printed command.")
        return 0

    # --review: interactive browser review loop (A propose → B edit → C crop + log)
    if args.review:
        from .review_server import run_review
        run_review(files, outdir, backend=args.backend,
                   display_max=args.display_max,
                   review_threshold=args.review_threshold,
                   out_format=args.format, quality=args.quality,
                   beside=beside, force=args.force,
                   only_flagged=args.only_flagged)
        return 0

    # --corners: manual single-image mode
    if args.corners is not None:
        if len(files) != 1:
            print("framefit: --corners works on a single image", file=sys.stderr)
            return 2
        from . import io
        from .pipeline import process_manual
        f = files[0]
        dst = _dst_for(f, outdir, beside, args.format)
        if dst.exists() and not args.force:
            print(f"framefit: {dst} exists — use --force to overwrite", file=sys.stderr)
            return 2
        image = io.load_bgr(f)
        r = process_manual(image, _parse_corners(args.corners), refine=args.refine)
        dst.parent.mkdir(parents=True, exist_ok=True)
        io.save_bgr(r.image, dst, quality=args.quality)
        print(f"[manual] {f.name} -> {dst}  (AR {r.aspect_ratio:.2f})")
        return 0

    backend = get_backend(args.backend)  # instantiate once (loads model if any)
    decided = set()
    if args.skip_decided:
        from . import feedback
        decided = feedback.decided_hashes()
    ok_count = 0
    rows = []  # (source, output, ok, ar, score, review)
    dst_dirs = set()  # directories crops were written to (for --beside report placement)
    for f in files:
        if decided:
            from . import feedback
            if feedback.sha1_of_file(f) in decided:
                print(f"[reviewed] {f.name}: already hand-decided — skipped")
                rows.append((str(f), "", 0, 0.0, 0.0, "-", "REVIEWED"))
                continue
        dst = _dst_for(f, outdir, beside, args.format)
        out_name = dst.name
        dst_dirs.add(dst.parent)
        if dst.exists() and not args.force:
            print(f"[skip] {f.name}: {dst.name} exists (use --force)", file=sys.stderr)
            rows.append((str(f), out_name, 0, 0.0, 0.0, "-", "SKIP"))
            continue
        try:
            r = process_file(f, dst, backend=backend, inset=args.inset,
                             expand=args.expand, detect_max=args.detect_max,
                             refine=args.refine, quality=args.quality)
        except Exception as e:  # noqa
            print(f"[error] {f.name}: {e}", file=sys.stderr)
            rows.append((str(f), "", 0, 0.0, 0.0, "-", "ERROR"))
            continue
        if r.ok:
            ok_count += 1
            # low_confidence (DL fell back to classic) is the reliable cut signal;
            # a low aspect score is a secondary flag.
            review = r.low_confidence or r.aspect_score < args.review_threshold
            status = "REVIEW" if review else "ok"
            rows.append((str(f), out_name, 1, r.aspect_ratio, r.aspect_score,
                         r.backend, status))
            reason = " [low-confidence: DL fell back]" if r.low_confidence else ""
            print(f"[ok]   {f.name} -> {dst}  (AR {r.aspect_ratio:.2f}, "
                  f"score {r.aspect_score:.2f}, {r.backend})"
                  f"{'  ⚠ REVIEW' + reason if review else ''}")
        else:
            rows.append((str(f), "", 0, 0.0, 0.0, "-", "MISS"))
            print(f"[miss] {f.name}: no slide detected", file=sys.stderr)

    # QA report (#2): per-file score + review flag.
    # --beside creates no output dir of its own, so drop the report beside the
    # sources (the common parent of where crops were written) instead of forcing
    # an arbitrary framefit_out/ into existence.
    review_n = sum(1 for r in rows if r[6] in ("REVIEW", "MISS", "ERROR"))
    if beside:
        report_dir = (Path(os.path.commonpath([str(d) for d in dst_dirs]))
                      if dst_dirs else Path.cwd())
    else:
        report_dir = outdir
    if rows:
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "framefit_report.tsv", "w") as rep:
            rep.write("source\toutput\tok\taspect_ratio\tscore\tbackend\tstatus\n")
            for src, out, ok, ar, sc, be, st in rows:
                rep.write(f"{src}\t{out}\t{ok}\t{ar:.3f}\t{sc:.3f}\t{be}\t{st}\n")

    print(f"\nframefit: {ok_count}/{len(files)} succeeded ({backend.name} backend)")
    if review_n:
        print(f"         {review_n} flagged for review — see {report_dir/'framefit_report.tsv'}")
        print(f"         fix a flagged image manually:  framefit \"<img>\" --pick -o {report_dir}")
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
