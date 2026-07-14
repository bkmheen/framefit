"""Phase 1 — offline detector evaluation against the human review log.

Ground truth = the corners a human confirmed/edited in the review UI (``final_quad``
in the learning log). For every logged image this replays a candidate detector and
scores its proposed quad against that human label, so any detector change can be
judged by a number instead of by eye.

Metrics (per image, aggregated):
  - **IoU**       polygon intersection-over-union of proposed vs human quad.
  - **delta_norm**  max corner move / image diagonal (same signal the log stores);
                    <0.02 ~ "clean", any edge >0.05 ~ "cut/blown".
  - **top_dnorm**  vertical error of the TOP edge / diagonal — the dominant failure
                   axis found on the OpticaImageSensorCongress2026 set.

Usage:
  .venv/bin/python research/eval_against_log.py                 # backend=auto, all logged
  .venv/bin/python research/eval_against_log.py -b classic      # score a specific backend
  .venv/bin/python research/eval_against_log.py --filter a/     # only source paths matching
  .venv/bin/python research/eval_against_log.py --worst 5 --csv out.csv

Exit status is 0 always; this is a report, not a gate. (The pytest regression gate
in Phase 4 will import :func:`evaluate` and assert on the aggregate.)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as a plain script (research/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from framefit import feedback, io                       # noqa: E402
from framefit.geometry import order_corners             # noqa: E402
from framefit.pipeline import process_image             # noqa: E402


def _iou(quad_a: np.ndarray, quad_b: np.ndarray, canvas: int = 1000) -> float:
    """IoU of two quads, rasterized on a shared normalized canvas."""
    both = np.vstack([quad_a, quad_b])
    lo = both.min(axis=0)
    span = float((both.max(axis=0) - lo).max()) or 1.0
    s = (canvas - 2) / span

    def _mask(q):
        m = np.zeros((canvas, canvas), np.uint8)
        pts = ((np.asarray(q, np.float32) - lo) * s + 1).astype(np.int32)
        cv2.fillConvexPoly(m, pts, 1)
        return m

    ma, mb = _mask(quad_a), _mask(quad_b)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return inter / union if union else 0.0


def _load_and_scale(rec: dict):
    """Return (bgr, scale_to_fullres) for a logged record.

    Prefer the untouched source photo (what production sees); fall back to the
    downscaled dataset copy. ``scale_to_fullres`` maps detector output (in the
    loaded image's coords) back to the full-res coordinate system that ``final_quad``
    / ``auto_quad`` live in, so metrics are comparable across both load paths.
    """
    fw = rec.get("image_width")
    src = rec.get("source_path")
    if src and Path(src).exists():
        bgr = io.load_bgr(src)
        return bgr, 1.0
    if rec.get("dataset_original"):
        p = feedback.review_root() / rec["dataset_original"]
        if p.exists():
            bgr = io.load_bgr(p)
            scale = fw / bgr.shape[1] if fw and bgr.shape[1] else 1.0
            return bgr, scale
    return None, None


def evaluate(backend: str = "auto", filt: str | None = None) -> list[dict]:
    """Score ``backend`` on every log record that has a human ``final_quad``.

    Detection goes through the production :func:`process_image` (same
    ``detect_max`` downscale and quad up-scaling the review server used), so the
    proposal matches what production would emit — not a raw ``detect()`` on a
    different-resolution image.
    """
    out = []
    for rec in feedback.read_log():
        final = rec.get("final_quad")
        if not final:                       # skipped image — no ground truth
            continue
        if filt and filt not in (rec.get("source_path") or ""):
            continue
        bgr, scale = _load_and_scale(rec)
        if bgr is None:
            out.append({"name": rec.get("source_name"), "status": "no-image"})
            continue

        gt = order_corners(np.asarray(final, np.float32))
        diag = float(rec.get("image_diagonal_px")
                     or np.hypot(*bgr.shape[1::-1]) * scale)

        # Reuse the production path (expand/inset/refine off — we score the raw
        # detected quad, not the post-processed crop). ``result.quad`` is in the
        # loaded image's coords; ``scale`` lifts it to full-res GT coords.
        res = process_image(bgr, backend=backend, expand=0.0, inset=0.0,
                            refine=False)
        if not res.ok or res.quad is None:
            out.append({"name": rec.get("source_name"), "status": "miss",
                        "iou": 0.0, "delta_norm": None, "top_dnorm": None,
                        "human_action": rec.get("action")})
            continue
        pred = order_corners(np.asarray(res.quad, np.float32) * scale)

        d = np.linalg.norm(pred - gt, axis=1)
        delta_norm = float(d.max() / diag) if diag else None
        top_dnorm = float(abs(((pred[0, 1] + pred[1, 1])
                               - (gt[0, 1] + gt[1, 1])) / 2) / diag) if diag else None
        out.append({
            "name": rec.get("source_name"),
            "status": "ok",
            "iou": round(_iou(pred, gt), 4),
            "delta_norm": round(delta_norm, 4) if delta_norm is not None else None,
            "top_dnorm": round(top_dnorm, 4) if top_dnorm is not None else None,
            "human_action": rec.get("action"),
            "backend_used": rec.get("backend"),
            "auto_low_conf": rec.get("auto_low_confidence"),
        })
    return out


def _summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("status") == "ok"]
    ious = [r["iou"] for r in scored]
    dns = [r["delta_norm"] for r in scored if r["delta_norm"] is not None]
    n = len(rows)
    return {
        "records": n,
        "scored": len(scored),
        "miss": sum(1 for r in rows if r.get("status") == "miss"),
        "no_image": sum(1 for r in rows if r.get("status") == "no-image"),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "median_iou": round(float(np.median(ious)), 4) if ious else 0.0,
        "clean_pct": round(100 * sum(1 for x in dns if x < 0.02) / len(dns), 1) if dns else 0.0,
        "cut_pct": round(100 * sum(1 for x in dns if x > 0.05) / len(dns), 1) if dns else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-b", "--backend", default="auto",
                    choices=["auto", "classic", "docaligner"])
    ap.add_argument("--filter", default=None,
                    help="only score records whose source_path contains this substring")
    ap.add_argument("--worst", type=int, default=8, help="list N worst by IoU")
    ap.add_argument("--csv", default=None, help="also write per-image rows to this CSV")
    args = ap.parse_args(argv)

    rows = evaluate(args.backend, args.filter)
    if not rows:
        print("eval: no logged records with a human final_quad "
              "(run some reviews first).", file=sys.stderr)
        return 0

    s = _summarize(rows)
    print(f"\n=== framefit detector eval  (backend={args.backend}"
          f"{', filter=' + args.filter if args.filter else ''}) ===")
    print(f"  records={s['records']}  scored={s['scored']}  "
          f"miss={s['miss']}  no_image={s['no_image']}")
    print(f"  mean IoU   = {s['mean_iou']:.3f}   median IoU = {s['median_iou']:.3f}")
    print(f"  clean(<.02)= {s['clean_pct']:.0f}%   cut(>.05)  = {s['cut_pct']:.0f}%")

    scored = [r for r in rows if r.get("status") == "ok"]
    worst = sorted(scored, key=lambda r: r["iou"])[:args.worst]
    if worst:
        print(f"\n  worst {len(worst)} by IoU:")
        print(f"  {'image':>16} {'IoU':>6} {'dNorm':>7} {'topDN':>7} {'human':>8} {'lowconf':>7}")
        for r in worst:
            print(f"  {str(r['name']):>16} {r['iou']:>6.3f} "
                  f"{(r['delta_norm'] if r['delta_norm'] is not None else float('nan')):>7.3f} "
                  f"{(r['top_dnorm'] if r['top_dnorm'] is not None else float('nan')):>7.3f} "
                  f"{str(r['human_action']):>8} {str(r['auto_low_conf']):>7}")

    if args.csv:
        keys = ["name", "status", "iou", "delta_norm", "top_dnorm",
                "human_action", "backend_used", "auto_low_conf"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
