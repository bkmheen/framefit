"""Review-signal report — the management/analysis surface over the decision log.

Every review decision is labeled (:func:`framefit.feedback.classify_review_signal`)
with how the detector's confidence call related to what the human actually did. Two
labels are the detector's *mistakes* and the reason this dataset exists:

  under_flag  the system was confident (auto-accepted, or shown as "good") yet the
              human had to change the crop — it should have asked for review.
  over_flag   the system asked for review but the human changed nothing — it should
              not have asked.

This module aggregates those labels so the "does this detection need review?" gate
can be calibrated/trained. It is UI-free and reads only the JSONL log shards.

Usage::

    python -m framefit.signals                 # summary counts
    python -m framefit.signals --errors        # list the under_flag / over_flag cases
    python -m framefit.signals --tsv out.tsv    # dump every labeled row (classifier input)
"""
from __future__ import annotations

import argparse
import sys

from . import feedback

# Columns dumped to TSV — detector confidence signals paired with the outcome label,
# i.e. exactly the (features, target) a review-gate classifier trains on.
_TSV_COLS = [
    "review_signal", "source_name", "source_sha1", "verdict_level", "was_flagged",
    "presented", "was_auto_accepted", "action", "was_modified", "revised",
    "prior_was_auto_accepted", "backend", "auto_low_confidence", "auto_aspect_score",
    "auto_detect_score", "max_corner_delta_px", "delta_norm",
]

_ERROR_SIGNALS = ("under_flag", "over_flag")


def _fmt_counts(counts: dict) -> str:
    total = sum(counts.values()) or 1
    order = ["under_flag", "over_flag", "correct_flag", "confirmed_pass",
             "auto_pass", "skip"]
    lines = []
    for k in order:
        n = counts.get(k, 0)
        mark = "  ⚠" if k in _ERROR_SIGNALS and n else ""
        lines.append(f"  {k:<15} {n:>5}  ({100*n/total:4.1f}%){mark}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="framefit.signals",
        description="Report review-signal labels (over-flag / under-flag) from the "
                    "decision log for calibration/analysis.")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="aspect-score review threshold used to reconstruct the flag "
                         "on legacy v2 records (default 0.90)")
    ap.add_argument("--errors", action="store_true",
                    help="list only the detector's mistakes (under_flag / over_flag)")
    ap.add_argument("--tsv", metavar="PATH",
                    help="write every labeled row to a TSV (classifier input)")
    args = ap.parse_args(argv)

    report = feedback.review_signals(review_threshold=args.threshold)
    counts, rows = report["counts"], report["rows"]
    total = sum(counts.values())

    print(f"framefit review signals — {total} decision(s) in {feedback.log_dir()}")
    print(_fmt_counts(counts))
    err = counts.get("under_flag", 0) + counts.get("over_flag", 0)
    if total:
        print(f"\n  detector review-gate errors: {err}/{total} "
              f"({100*err/total:.1f}%)")

    if args.errors:
        bad = [r for r in rows if r["review_signal"] in _ERROR_SIGNALS]
        print(f"\n{len(bad)} flagged-gate error(s):")
        for r in bad:
            print(f"  [{r['review_signal']:<10}] {r.get('source_name')}  "
                  f"verdict={r.get('verdict_level')} action={r.get('action')} "
                  f"aspect={r.get('auto_aspect_score')} "
                  f"maxΔpx={r.get('max_corner_delta_px')}")

    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as f:
            f.write("\t".join(_TSV_COLS) + "\n")
            for r in rows:
                f.write("\t".join("" if r.get(c) is None else str(r.get(c))
                                  for c in _TSV_COLS) + "\n")
        print(f"\nwrote {len(rows)} labeled row(s) -> {args.tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
