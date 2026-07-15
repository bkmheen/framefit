"""Headless decision logger + self-contained learning dataset for the review loop.

Every review decision — whether the auto-detected corners were **accepted**
unchanged, **modified**, drawn **from scratch** (auto-detect failed), or the image
was **skipped** — is appended to a per-host JSONL shard and, optionally, the
original + rectified crop are copied into a content-addressed dataset. The point
is later *analysis and learning*: correlate the detector's own confidence signals
(backend, low-confidence fallback, aspect score) against how much the human had to
move the corners (``delta_norm``), so the confidence flag / review threshold can be
calibrated and a corrector eventually trained.

Design constraints:
- **Sync-safe.** Data lives under ``~/MarvisHome`` (synced across Macs). Two Macs
  appending to one shared file would produce "conflicted copy" collisions, so each
  host writes ONLY its own ``log/log-<host>.jsonl``; reads glob every shard.
- **Dedupe by content.** The image's SHA-1 is its stable identity across renames
  and machines; the same image is never stored or logged twice.
- **UI-free.** Nothing here imports the server or opens a browser, so the whole
  dataset can be regenerated offline (see :mod:`framefit.batch_replay`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from . import __version__, io
from .geometry import aspect_score_wh, order_corners

SCHEMA_VERSION = 3  # v3: review-signal labels — distinguish over-flag vs under-flag
                    #     (see review_signals()); v2 added auto_detect_score.

# The review-signal vocabulary. These labels are the training target for a future
# "does this detection actually need human review?" classifier. Two of them are the
# detector's *mistakes* and the whole point of collecting this dataset:
#   under_flag — the system was confident (auto-accepted, or shown as "good") yet the
#                human had to change the crop. It SHOULD have asked. (사용자가 물어보지
#                않은 것을 다시 수정)  ← false negative of the review gate.
#   over_flag  — the system asked for review (flagged suspect/fail) but the human made
#                no change. It should NOT have asked. (확인 필요라 했는데 그냥 넘어감)
#                ← false positive of the review gate.
# The rest are correct outcomes or non-signal:
#   correct_flag   — flagged and the human did change it (review was warranted).
#   confirmed_pass — shown as good and left unchanged (confident and correct).
#   auto_pass      — auto-accepted and never revised (confident and correct).
#   skip           — discarded; carries no flag-accuracy signal.
REVIEW_SIGNALS = ("under_flag", "over_flag", "correct_flag",
                  "confirmed_pass", "auto_pass", "skip")


def classify_review_signal(*, action: str, was_flagged: bool, was_modified: bool,
                           was_auto_accepted: bool, revised: bool,
                           prior_was_auto_accepted: bool) -> str:
    """Single source of truth mapping one decision's atomic facts to a
    :data:`REVIEW_SIGNALS` label. Used both when a record is written and by
    :func:`review_signals` at analysis time, so the two never drift."""
    if action == "skip":
        return "skip"
    # Signal A (false negative): confident but the crop actually needed changing —
    # either an auto-accept the human came back to fix, or a "good" the human still
    # edited.
    if (revised and prior_was_auto_accepted) or (not was_flagged and was_modified):
        return "under_flag"
    if was_flagged and was_modified:
        return "correct_flag"
    # Signal B (false positive): flagged for review but the human changed nothing.
    if was_flagged and not was_modified:
        return "over_flag"
    if was_auto_accepted:
        return "auto_pass"
    return "confirmed_pass"

# Longest side (px) for the stored original copy — bounds synced-folder growth
# while keeping enough resolution to re-inspect a bad crop. Override with env.
ORIGINAL_MAX_SIDE = int(os.environ.get("FRAMEFIT_ORIG_MAX", "2560"))


def review_root() -> Path:
    """Root of the shared learning dataset. Override with ``FRAMEFIT_REVIEW_DIR``.

    Defaults under ``~/MarvisHome`` (a folder synced across the user's Macs), so
    the dataset compounds no matter which machine did the reviewing.
    """
    env = os.environ.get("FRAMEFIT_REVIEW_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "MarvisHome" / "Code" / "framefit" / "reviews"


def _host() -> str:
    return (socket.gethostname().split(".")[0] or "unknown").strip()


def log_dir() -> Path:
    return review_root() / "log"


def shard_path() -> Path:
    """This host's append-only JSONL shard."""
    return log_dir() / f"log-{_host()}.jsonl"


def originals_dir() -> Path:
    return review_root() / "originals"


def crops_dir() -> Path:
    return review_root() / "crops"


# --------------------------------------------------------------------------- #
# Hashing / identity
# --------------------------------------------------------------------------- #
def sha1_of_file(path: str | Path, _buf: int = 1 << 20) -> str:
    """Content SHA-1 — the image's stable identity used for dedupe."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def read_log() -> list[dict]:
    """Every decision across all host shards, deduped on ``source_sha1``
    (last write wins)."""
    by_sha: dict[str, dict] = {}
    d = log_dir()
    if not d.is_dir():
        return []
    for shard in sorted(d.glob("log-*.jsonl")):
        try:
            for line in shard.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("source_sha1")
                if key:
                    by_sha[key] = rec
        except OSError:
            continue
    return list(by_sha.values())


def decided_hashes() -> set[str]:
    """SHA-1s already decided (any host) — used to skip work on resume."""
    return {r["source_sha1"] for r in read_log() if r.get("source_sha1")}


# --------------------------------------------------------------------------- #
# Record building
# --------------------------------------------------------------------------- #
def _quad_list(quad) -> Optional[list[list[float]]]:
    if quad is None:
        return None
    arr = np.asarray(quad, dtype=np.float64)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in arr]


def build_record(
    *,
    source_path: str | Path,
    source_sha1: str,
    image_width: int,
    image_height: int,
    backend: str,
    auto_low_confidence: bool,
    auto_aspect_score: float,
    auto_quad,                 # full-res px or None
    final_quad,                # full-res px or None (None only when skipped)
    action: str,               # accept | modify | skip | manual_from_scratch
    display_scale: float,
    auto_detect_score: Optional[float] = None,  # classical composite score (calibration)
    output_path: Optional[str | Path],
    output_aspect_ratio: float = 0.0,
    output_aspect_score: float = 0.0,
    dataset_original: Optional[str] = None,
    dataset_crop: Optional[str] = None,
    # -- review-signal context (v3) -------------------------------------- #
    verdict_level: str = "good",        # good | suspect | fail (self-assessment)
    verdict_reason: str = "",           # why the verdict was reached
    presented: bool = True,             # was the image actually shown to the human?
    was_auto_accepted: bool = False,    # silently cropped (--only-flagged good)
    revised: bool = False,              # this decision overrides an earlier one
    prior_action: Optional[str] = None,        # the action being overridden
    prior_was_auto_accepted: bool = False,     # ...and was it an auto-accept?
) -> dict:
    """Assemble one decision record, computing the calibration-relevant deltas and
    the :func:`classify_review_signal` label that distinguishes over-flag / under-flag
    mistakes for later analysis."""
    diag = math.hypot(image_width, image_height)
    auto_ordered = None if auto_quad is None else order_corners(
        np.asarray(auto_quad, dtype=np.float32))
    final_ordered = None if final_quad is None else order_corners(
        np.asarray(final_quad, dtype=np.float32))

    per_corner = None
    max_d = mean_d = delta_norm = None
    if auto_ordered is not None and final_ordered is not None:
        d = np.linalg.norm(final_ordered - auto_ordered, axis=1)
        per_corner = [round(float(v), 2) for v in d]
        max_d = round(float(d.max()), 2)
        mean_d = round(float(d.mean()), 2)
        delta_norm = round(float(d.max() / diag), 5) if diag else None

    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "tool_version": __version__,
        "host": _host(),
        "source_path": str(Path(source_path).resolve()),
        "source_name": Path(source_path).name,
        "source_sha1": source_sha1,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_diagonal_px": round(diag, 2),
        "backend": backend,
        "auto_low_confidence": bool(auto_low_confidence),
        "auto_aspect_score": round(float(auto_aspect_score), 4),
        "auto_detect_score": (None if auto_detect_score is None
                              else round(float(auto_detect_score), 4)),
        "auto_quad": _quad_list(auto_ordered),
        "final_quad": _quad_list(final_ordered),
        "action": action,
        "was_modified": (was_modified := action in ("modify", "manual_from_scratch")),
        # -- review-signal context + derived label (v3) ------------------ #
        "verdict_level": verdict_level,
        "verdict_reason": verdict_reason,
        "was_flagged": (was_flagged := verdict_level != "good"),
        "presented": bool(presented),
        "was_auto_accepted": bool(was_auto_accepted),
        "revised": bool(revised),
        "prior_action": prior_action,
        "prior_was_auto_accepted": bool(prior_was_auto_accepted),
        "review_signal": classify_review_signal(
            action=action, was_flagged=was_flagged, was_modified=was_modified,
            was_auto_accepted=was_auto_accepted, revised=revised,
            prior_was_auto_accepted=prior_was_auto_accepted),
        "display_scale": round(float(display_scale), 6),
        "output_path": None if output_path is None else str(output_path),
        "output_aspect_ratio": round(float(output_aspect_ratio), 4),
        "output_aspect_score": round(float(output_aspect_score), 4),
        "per_corner_delta_px": per_corner,
        "max_corner_delta_px": max_d,
        "mean_corner_delta_px": mean_d,
        "delta_norm": delta_norm,
        "dataset_original": dataset_original,
        "dataset_crop": dataset_crop,
    }


# --------------------------------------------------------------------------- #
# Dataset storage (content-addressed, deduped)
# --------------------------------------------------------------------------- #
def _downscale(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def store_dataset(
    source_sha1: str,
    original_bgr: np.ndarray,
    crop_bgr: Optional[np.ndarray],
) -> tuple[Optional[str], Optional[str]]:
    """Copy the original (downscaled) and the rectified crop into the dataset,
    keyed by SHA-1. Skips any file that already exists (dedupe). Returns the
    dataset-relative paths (or None on failure)."""
    root = review_root()
    orig_rel = crop_rel = None
    try:
        originals_dir().mkdir(parents=True, exist_ok=True)
        op = originals_dir() / f"{source_sha1}.jpg"
        if not op.exists():
            io.save_bgr(_downscale(original_bgr, ORIGINAL_MAX_SIDE), op, quality=92)
        orig_rel = str(op.relative_to(root))
    except (OSError, ValueError):
        orig_rel = None
    if crop_bgr is not None:
        try:
            crops_dir().mkdir(parents=True, exist_ok=True)
            cp = crops_dir() / f"{source_sha1}.jpg"
            if not cp.exists():
                io.save_bgr(crop_bgr, cp, quality=95)
            crop_rel = str(cp.relative_to(root))
        except (OSError, ValueError):
            crop_rel = None
    return orig_rel, crop_rel


def append_record(record: dict) -> Path:
    """Append one record to this host's shard (creates dirs as needed)."""
    p = shard_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p


def _signal_for_record(rec: dict, review_threshold: float = 0.90) -> str:
    """The review-signal label for one deduped record. Uses the stored ``review_signal``
    when present (v3+); otherwise reconstructs it from whatever a v2 record carried so
    old data still contributes to the analysis."""
    if rec.get("review_signal") in REVIEW_SIGNALS:
        return rec["review_signal"]
    action = rec.get("action", "")
    was_modified = rec.get("was_modified", action in ("modify", "manual_from_scratch"))
    # v2 had no verdict; reconstruct the flag from the raw auto signals.
    if "was_flagged" in rec:
        was_flagged = bool(rec["was_flagged"])
    else:
        has_quad = rec.get("auto_quad") is not None
        was_flagged = ((not has_quad) or bool(rec.get("auto_low_confidence"))
                       or float(rec.get("auto_aspect_score", 0.0)) < review_threshold)
    return classify_review_signal(
        action=action, was_flagged=was_flagged, was_modified=was_modified,
        was_auto_accepted=bool(rec.get("was_auto_accepted", False)),
        revised=bool(rec.get("revised", False)),
        prior_was_auto_accepted=bool(rec.get("prior_was_auto_accepted", False)))


def review_signals(review_threshold: float = 0.90) -> dict:
    """Management/analysis view over the whole decision log — the basis for building
    a "does this detection need review?" classifier.

    Returns ``{"counts": {label: n, ...}, "rows": [per-image dicts]}`` where each row
    pairs the detector's confidence signals (backend, low-confidence fallback, aspect
    & detect scores, corner delta) with the outcome label. The two error labels
    (:data:`under_flag`, :data:`over_flag`) are the mislabeled cases a calibrated gate
    must eliminate; filter on them to train or evaluate."""
    rows: list[dict] = []
    counts: dict[str, int] = {s: 0 for s in REVIEW_SIGNALS}
    for rec in read_log():
        signal = _signal_for_record(rec, review_threshold)
        counts[signal] = counts.get(signal, 0) + 1
        rows.append({
            "source_sha1": rec.get("source_sha1"),
            "source_name": rec.get("source_name"),
            "review_signal": signal,
            "verdict_level": rec.get("verdict_level"),
            "was_flagged": rec.get("was_flagged"),
            "presented": rec.get("presented"),
            "was_auto_accepted": rec.get("was_auto_accepted"),
            "action": rec.get("action"),
            "was_modified": rec.get("was_modified"),
            "revised": rec.get("revised"),
            "prior_was_auto_accepted": rec.get("prior_was_auto_accepted"),
            # detector confidence signals (classifier features)
            "backend": rec.get("backend"),
            "auto_low_confidence": rec.get("auto_low_confidence"),
            "auto_aspect_score": rec.get("auto_aspect_score"),
            "auto_detect_score": rec.get("auto_detect_score"),
            "max_corner_delta_px": rec.get("max_corner_delta_px"),
            "delta_norm": rec.get("delta_norm"),
        })
    return {"counts": counts, "rows": rows}


def confidence_verdict(low_confidence: bool, aspect_score: float,
                       has_quad: bool, review_threshold: float = 0.90) -> dict:
    """Turn the detector's raw signals into a human-facing self-assessment —
    the 'is this detection likely right or wrong?' estimate shown in Module A."""
    if not has_quad:
        return {"level": "fail", "label": "자동검출 실패 — 직접 4점 지정",
                "reason": "no quad"}
    if low_confidence:
        return {"level": "suspect", "label": "의심 — DL 폴백(저신뢰)",
                "reason": "low_confidence"}
    if aspect_score < review_threshold:
        return {"level": "suspect",
                "label": f"의심 — 종횡비 점수 낮음({aspect_score:.2f})",
                "reason": "low_aspect_score"}
    return {"level": "good", "label": f"양호(점수 {aspect_score:.2f})",
            "reason": "ok"}
