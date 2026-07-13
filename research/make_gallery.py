"""Build a final before/after gallery of the installed framefit package on the
local sample set. Outputs research/out_final/ (gitignored).

Usage:  python research/make_gallery.py [--backend auto] [--inset 0.0]
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2

import framefit
from framefit import io
from framefit.backends import get_backend

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = Path(__file__).resolve().parent / "out_final"
OUT.mkdir(exist_ok=True)


def thumb_b64(bgr, max_side=460):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--inset", type=float, default=0.0)
    args = ap.parse_args()

    backend = get_backend(args.backend)
    samples = sorted(SAMPLES.glob("*.HEIC"))
    rows = []
    ok = 0
    for sp in samples:
        img = io.load_bgr(sp)
        r = framefit.process_image(img, backend=backend, inset=args.inset)
        before = thumb_b64(img)
        if r.ok:
            ok += 1
            io.save_bgr(r.image, OUT / f"{sp.stem}_framefit.jpg")
            after = thumb_b64(r.image)
            meta = f"AR {r.aspect_ratio:.2f} · score {r.aspect_score:.2f} · {r.backend}"
        else:
            after, meta = None, "NO DETECTION"
        rows.append((sp.stem, before, after, meta))
        print(f"{sp.stem:14s} {'ok' if r.ok else 'MISS':4s} {meta}")

    html = [
        "<!doctype html><meta charset='utf-8'><title>framefit final gallery</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}"
        "h1{font-weight:600}.row{display:flex;gap:14px;align-items:center;"
        "border-bottom:1px solid #333;padding:12px 0}.row>div{flex:0 0 auto}"
        ".lab{width:120px;color:#9cf}img{max-width:460px;display:block;border:1px solid #333}"
        ".meta{color:#4c8;width:230px}.arrow{color:#666;font-size:24px}</style>",
        f"<h1>framefit — final gallery ({args.backend} backend, inset={args.inset})</h1>",
        f"<p>{ok}/{len(samples)} succeeded. Left = original photo, right = framefit output.</p>",
    ]
    for stem, before, after, meta in rows:
        a_img = (f"<img src='data:image/jpeg;base64,{after}'>" if after
                 else "<span style='color:#e55'>FAIL</span>")
        html.append(
            f"<div class='row'><div class='lab'>{stem}</div>"
            f"<div><img src='data:image/jpeg;base64,{before}'></div>"
            f"<div class='arrow'>&rarr;</div><div>{a_img}</div>"
            f"<div class='meta'>{meta}</div></div>"
        )
    (OUT / "gallery.html").write_text("".join(html), encoding="utf-8")
    print(f"\n{ok}/{len(samples)} ok  ->  {OUT/'gallery.html'}")


if __name__ == "__main__":
    main()
