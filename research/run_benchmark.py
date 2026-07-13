"""Run candidate detectors over the HEIC sample set and build a comparison report.

Outputs (all under research/out/, gitignored):
  - <stem>__<candidate>.jpg     : perspective-corrected + cropped slide
  - <stem>__<candidate>_overlay.jpg : input (downscaled) with detected quad drawn
  - report.html                 : side-by-side grid of every sample x candidate
  - results.csv                 : per (sample, candidate) success + metrics

Usage:
  python research/run_benchmark.py
"""
from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pillow_heif

from detectors import DETECTORS, order_corners, quad_area

pillow_heif.register_heif_opener()

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

DETECT_MAX = 1400  # longest side used for detection (speed); warp uses full res


def load_heic_bgr(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def warp_from_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    W = int(max(wA, wB))
    H = int(max(hA, hB))
    W = max(W, 10)
    H = max(H, 10)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def to_jpg_bytes(bgr: np.ndarray, max_side=520) -> bytes:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()


def b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()


def main():
    samples = sorted(SAMPLES.glob("*.HEIC"))
    cand_names = list(DETECTORS.keys())
    rows = []
    # cell_imgs[stem][cand] = (overlay_b64, warp_b64 or None, info)
    cells = {}

    for sp in samples:
        stem = sp.stem
        full = load_heic_bgr(sp)
        H, W = full.shape[:2]
        s = DETECT_MAX / max(H, W)
        small = cv2.resize(full, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        inv = 1.0 / s
        cells[stem] = {}
        for name, fn in DETECTORS.items():
            t0 = time.time()
            try:
                quad_small = fn(small)
            except Exception as e:  # noqa
                quad_small = None
                err = str(e)
            else:
                err = ""
            dt = (time.time() - t0) * 1000

            overlay = small.copy()
            success = quad_small is not None
            area_frac = 0.0
            if success:
                area_frac = quad_area(quad_small) / (small.shape[0] * small.shape[1])
                cv2.polylines(overlay, [quad_small.astype(np.int32)], True, (0, 0, 255), 3)
                for p in quad_small.astype(int):
                    cv2.circle(overlay, tuple(p), 8, (0, 255, 0), -1)
                quad_full = order_corners(quad_small * inv)
                warp = warp_from_quad(full, quad_full)
                warp_b64 = b64(to_jpg_bytes(warp))
                cv2.imwrite(str(OUT / f"{stem}__{name}.jpg"), warp)
            else:
                warp_b64 = None
                cv2.putText(overlay, "FAIL", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.imwrite(str(OUT / f"{stem}__{name}_overlay.jpg"), overlay)
            cells[stem][name] = {
                "overlay": b64(to_jpg_bytes(overlay)),
                "warp": warp_b64,
                "ok": success,
                "area": area_frac,
                "ms": dt,
                "err": err,
            }
            rows.append([stem, name, int(success), f"{area_frac:.3f}",
                         f"{dt:.0f}", err])
            print(f"{stem:16s} {name:16s} ok={success} area={area_frac:.2f} {dt:.0f}ms {err}")

    # CSV
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "candidate", "success", "area_frac", "ms", "error"])
        w.writerows(rows)

    write_report(samples, cand_names, cells)
    # summary
    print("\n=== SUCCESS RATE ===")
    for name in cand_names:
        n = sum(1 for sp in samples if cells[sp.stem][name]["ok"])
        print(f"  {name:16s} {n}/{len(samples)}")


def write_report(samples, cand_names, cells):
    parts = ["<!doctype html><meta charset='utf-8'><title>framefit candidate benchmark</title>",
             "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}",
             "table{border-collapse:collapse}td,th{border:1px solid #333;padding:4px;vertical-align:top;text-align:center}",
             "img{display:block;max-width:260px;height:auto;margin:2px auto}",
             ".ok{color:#4c8}.fail{color:#e55}small{color:#999}</style>"]
    parts.append("<h1>framefit — candidate detection benchmark</h1>")
    parts.append(f"<p>{len(samples)} samples &times; {len(cand_names)} candidates. "
                 "Top row per cell = detected quad overlay; bottom = warped+cropped output.</p>")
    parts.append("<table><tr><th>sample</th>")
    for name in cand_names:
        parts.append(f"<th>{name}</th>")
    parts.append("</tr>")
    for sp in samples:
        stem = sp.stem
        parts.append(f"<tr><th>{stem}</th>")
        for name in cand_names:
            c = cells[stem][name]
            cls = "ok" if c["ok"] else "fail"
            status = f"area {c['area']*100:.0f}% · {c['ms']:.0f}ms" if c["ok"] else "FAIL"
            cell = f"<td><span class='{cls}'>{status}</span>"
            cell += f"<img src='data:image/jpeg;base64,{c['overlay']}'>"
            if c["warp"]:
                cell += f"<img src='data:image/jpeg;base64,{c['warp']}'>"
            cell += "</td>"
            parts.append(cell)
        parts.append("</tr>")
    parts.append("</table>")
    (OUT / "report.html").write_text("".join(parts), encoding="utf-8")
    print(f"\nReport: {OUT/'report.html'}")


if __name__ == "__main__":
    main()
