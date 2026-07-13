"""Experiment: does tone preprocessing help the DL corner detector (C5)?

For each sample, run C5 (DocAligner) on each preprocessing variant (raw, A1, A2,
A3). Detection uses the preprocessed downscale; the warp is taken from the
untouched original. Emits a comparison grid + metrics.

Outputs under research/out_dl/ (gitignored):
  - <stem>__<variant>_overlay.jpg  : preprocessed detect-image + detected quad
  - <stem>__<variant>_warp.jpg     : warp/crop from the ORIGINAL using that quad
  - report_dl.html, results_dl.csv

Usage:  python research/run_dl_experiment.py
"""
from __future__ import annotations

import base64
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif

from detectors import order_corners, quad_area
from detectors_dl import detect_docaligner
from preprocess import PREPROCESSORS

pillow_heif.register_heif_opener()

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = Path(__file__).resolve().parent / "out_dl"
OUT.mkdir(exist_ok=True)
DETECT_MAX = 1400

# standard slide aspect ratios we score against
STD_ARS = [16 / 9, 16 / 10, 4 / 3, 3 / 2]


def load_heic_bgr(path: Path) -> np.ndarray:
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def warp_from_quad(img, quad):
    tl, tr, br, bl = quad
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    W, H = max(W, 10), max(H, 10)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def aspect_score(quad):
    """1.0 = perfectly matches a standard slide AR; lower = worse."""
    tl, tr, br, bl = quad
    W = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2
    H = (np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2
    if H < 1:
        return 0.0, 0.0
    ar = W / H
    best = min(abs(ar - s) / s for s in STD_ARS)
    return ar, max(0.0, 1.0 - best)


def to_jpg_b64(bgr, max_side=460):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode()


def main():
    samples = sorted(SAMPLES.glob("*.HEIC"))
    variants = list(PREPROCESSORS.keys())
    cells = {}
    rows = []

    for sp in samples:
        stem = sp.stem
        full = load_heic_bgr(sp)
        H, W = full.shape[:2]
        s = DETECT_MAX / max(H, W)
        small = cv2.resize(full, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        inv = 1.0 / s
        cells[stem] = {}
        for vname, vfn in PREPROCESSORS.items():
            det_img = vfn(small)
            t0 = time.time()
            try:
                quad = detect_docaligner(det_img)
            except Exception as e:  # noqa
                quad, err = None, str(e)
            else:
                err = ""
            dt = (time.time() - t0) * 1000

            overlay = det_img.copy()
            ok = quad is not None
            ar, arsc, areaf = 0.0, 0.0, 0.0
            if ok:
                areaf = quad_area(quad) / (small.shape[0] * small.shape[1])
                ar, arsc = aspect_score(quad)
                cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 0, 255), 3)
                for p in quad.astype(int):
                    cv2.circle(overlay, tuple(p), 7, (0, 255, 0), -1)
                warp = warp_from_quad(full, order_corners(quad * inv))
                cv2.imwrite(str(OUT / f"{stem}__{vname}_warp.jpg"), warp)
                warp_b64 = to_jpg_b64(warp)
            else:
                cv2.putText(overlay, "FAIL", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                warp_b64 = None
            cv2.imwrite(str(OUT / f"{stem}__{vname}_overlay.jpg"), overlay)
            cells[stem][vname] = {
                "overlay": to_jpg_b64(overlay), "warp": warp_b64, "ok": ok,
                "ar": ar, "arsc": arsc, "area": areaf, "ms": dt,
            }
            rows.append([stem, vname, int(ok), f"{ar:.3f}", f"{arsc:.2f}",
                         f"{areaf:.3f}", f"{dt:.0f}", err])
            print(f"{stem:14s} {vname:14s} ok={ok} AR={ar:.2f} arScore={arsc:.2f} "
                  f"area={areaf:.2f} {dt:.0f}ms {err}")

    with open(OUT / "results_dl.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "variant", "ok", "aspect", "ar_score", "area", "ms", "err"])
        w.writerows(rows)

    write_report(samples, variants, cells)
    print("\n=== mean aspect-ratio score (higher=closer to a real slide AR) ===")
    for v in variants:
        vals = [cells[sp.stem][v]["arsc"] for sp in samples if cells[sp.stem][v]["ok"]]
        n = sum(1 for sp in samples if cells[sp.stem][v]["ok"])
        m = sum(vals) / len(vals) if vals else 0
        print(f"  {v:14s} ok={n}/{len(samples)}  meanARscore={m:.2f}")


def write_report(samples, variants, cells):
    p = ["<!doctype html><meta charset='utf-8'><title>framefit C5 preprocessing</title>",
         "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}",
         "table{border-collapse:collapse}td,th{border:1px solid #333;padding:4px;text-align:center;vertical-align:top}",
         "img{display:block;max-width:240px;margin:2px auto}.ok{color:#4c8}.fail{color:#e55}</style>",
         "<h1>framefit — C5 (DocAligner) with tone preprocessing</h1>",
         "<p>Per cell: top = detected quad on the <b>preprocessed</b> detect-image; "
         "bottom = warp/crop from the <b>original</b>. AR = detected aspect ratio, "
         "arScore = closeness to a standard slide ratio (1.0 best).</p>",
         "<table><tr><th>sample</th>"]
    for v in variants:
        p.append(f"<th>{v}</th>")
    p.append("</tr>")
    for sp in samples:
        stem = sp.stem
        p.append(f"<tr><th>{stem}</th>")
        for v in variants:
            c = cells[stem][v]
            cls = "ok" if c["ok"] else "fail"
            label = (f"AR {c['ar']:.2f} · s{c['arsc']:.2f} · {c['ms']:.0f}ms"
                     if c["ok"] else "FAIL")
            cell = f"<td><span class='{cls}'>{label}</span>"
            cell += f"<img src='data:image/jpeg;base64,{c['overlay']}'>"
            if c["warp"]:
                cell += f"<img src='data:image/jpeg;base64,{c['warp']}'>"
            p.append(cell + "</td>")
        p.append("</tr>")
    p.append("</table>")
    (OUT / "report_dl.html").write_text("".join(p), encoding="utf-8")
    print(f"\nReport: {OUT/'report_dl.html'}")


if __name__ == "__main__":
    main()
