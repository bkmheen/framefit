# framefit — Detection Improvement Plan

*Data-driven plan to fix the auto-detector, grounded in the review log. Managed
in git; update the "Evidence" and "Status" sections as the dataset grows.*

Last updated: **2026-07-14** · First evidence set: OpticaImageSensorCongress2026 `a/` (11 shots).

---

## 1. What went wrong (evidence-based)

On the `a/` batch (11 HEIC), the human review corrected **7 of 11** crops. The
per-corner correction deltas (from the learning log, `final_quad − auto_quad`,
positive = final edge is lower/inward) isolate the failure to the **top edge**:

| img | action | delta_norm | top edge move | diagnosis |
|-----|--------|-----------:|--------------:|-----------|
| 3673 | modify | 0.251 | **−966 px** | title cut; detector grabbed only the bright interior image |
| 3674 | modify | 0.067 | **−453 px** | title/legend cut; same |
| 3676 | modify | 0.098 | **+561 px** | ceiling + lights swallowed as "slide" |
| 3677 | modify | 0.077 | **+253 px** | trapezoid; top-right wedge of ceiling |
| 3678 | modify | 0.089 | **+329 px** | ceiling included at top |
| 3682 | modify | 0.101 | +366 / bot +457 | shifted; bottom graph row lost |
| 3675 | modify | 0.008 | −9 px | negligible (essentially an accept) |
| 3679/80/81/83 | accept | 0.000 | 0 | correct as-is (DocAligner succeeded) |

Two opposite top-edge failure modes, **both from the same root**: the classical
detector reasons about *brightness*, not about *the slide as an object*.

### Root cause, by layer

1. **Detection algorithm — `backends/classic.py`.**
   Pipeline is `gray → GaussianBlur → global Otsu threshold → morph → largest
   convex 4-gon (approxPolyDP), else minAreaRect`. Global Otsu picks **one**
   brightness cut for the whole frame. On a color-cast projected slide:
   - the brightest region is the slide's *inner picture* (blue image in 3673,
     green render in 3674), while the **title band sits on a darker projector-
     black** → Otsu classifies the title as "background" → the quad locks onto the
     interior only, cutting the title (top pulled **down**, so the human pulls it
     **up**: 3673/3674).
   - in dim shots the **ceiling lights are as bright as the slide** → the bright
     blob merges slide+ceiling → top pulled **up** into the ceiling (3676/78/82).
   - `minAreaRect` fallback (hit by 3673) returns an axis-agnostic box that, after
     ordering/clamping, spills to image borders (3673 auto quad ran x=0…5708,
     y=1575…4280 — garbage).
   The detector never keys on the **projection screen's black rectangular frame**,
   which is actually the cleanest, most reliable cue in every one of these photos.

2. **Confidence gate — `backends/auto.py`.**
   `aspect_score ≥ min_score(0.80)` accepts a quad. Aspect score only asks "is this
   rectangle slide-shaped?", not "is this the *right* rectangle." 3673's wrong crop
   scored **0.81 and passed**. The one honest signal we have is `low_confidence`
   (DocAligner fell back) — it *flagged* all 7 bad ones, but nothing *fixed* them.

3. **Safety margin — `expand=0.04`.**
   A 4% outward grow cannot recover a title band that is ~25% of slide height
   (3673 delta_norm 0.25). The margin is tuned for pixel-level slop, not
   whole-region misses.

**Why DocAligner didn't save them:** it fell back (`low_confidence=True`) on every
corrected shot — heavy blue/green projector color-cast is out of its training
distribution. The 4 accepts are exactly the shots where DocAligner held.

---

## 2. Improvement process (staged, gated on the log)

The review log is the asset the whole review module was built to produce. Use it as
ground truth; never ship a detector change that regresses it.

### Phase 0 — Harvest (ongoing, in place)
Every human decision already appends `(auto_quad, final_quad, delta_norm, backend,
low_confidence, aspect_score)` to `~/MarvisHome/Code/framefit/reviews/`. Keep
running review passes on new decks to grow the labeled set. **11 pairs today**;
target ≥100 across ≥5 rooms/lighting before trusting aggregate metrics.

### Phase 1 — Offline eval harness ✅ (built & validated 2026-07-14)
`research/eval_against_log.py`: for every log record with a `final_quad`, reload the
image, run the **production `process_image`** path (same `detect_max` downscale +
quad up-scaling the review server used — faithfulness verified: re-run `pred` vs
logged `auto_quad` ≈ 0.000), and report against the human label:
- **IoU** of proposed vs final quad (polygon), and **delta_norm**;
- **top_dnorm** (vertical error of the TOP edge / diagonal) — the dominant axis;
- **% "clean"** (delta_norm < 0.02) and **% "cut"** (any edge moved > 0.05);
- worst-N table (name, IoU, dNorm, topDN, human action, low-confidence flag).

It prefers the untouched source photo, else the **downscaled dataset copy** — so it
keeps working after the source folder is deleted (validated: the `a/` set was
removed mid-work and eval still scored all 11 from the SHA-1 dataset copies).

**Baseline (backend=auto, `a/` set, n=11):**

| metric | value |
|--------|------:|
| mean IoU | **0.808** |
| median IoU | 0.845 |
| clean (<0.02) | 46% |
| cut (>0.05) | **54%** |

Worst 6 are all `modify`, and `top_dnorm` is the majority of each error (e.g. 3676:
dNorm 0.098, topDN 0.079). This is the number every change below must beat.

### Phase 2 — Detector fixes, ranked by the evidence (top edge first)
1. **Screen-frame detector (new hypothesis).** Detect the projection screen by its
   **dark rectangular bezel**, not by interior brightness: adaptive/edge path
   (`Canny → dilate → largest convex quad`) plus morphological **hole-filling** so
   an internally-dark title band never splits the region. Add as a sibling of
   `ClassicDetector`.
2. **Multi-hypothesis + honest scoring in `auto.py`.** Generate several candidates
   (Otsu-bright, edge-screen, dark-frame), and score by more than aspect: reward
   quads whose **border lies on a strong straight gradient** and whose **interior
   is more uniform than the immediate exterior** (slide vs wall). Pick the best;
   this is what separates "right rectangle" from "bright blob."
3. **CLAHE-before-detect for tinted shots.** ~~`preprocess.clahe` already exists;
   run detection on the local-contrast image…~~ **REJECTED by probe (2026-07-14).**
   Measured blanket CLAHE / gamma before the classic detector on the `a/` set:
   mean IoU **0.737 → 0.710 (CLAHE) / 0.688 (gamma)** — it nudges the interior-grab
   case up (3673 0.239→0.357, still a fail) but *degrades* the others. A global
   preprocessing swap is not the fix; brightness-thresholding is the limit. (Kept
   as an option only if gated per-image by a color-cast test — low priority.)
4. **Confidence-aware expand.** When `low_confidence`, raise `expand` (e.g. 0.04 →
   0.10) — cheap insurance that would have partially saved 3676/78 automatically.

### Phase 3 — Confidence calibration (turn delta_norm into the flag)
Cross-tabulate `delta_norm` (truth) against `low_confidence` + `aspect_score` to set
`--review-threshold` so "flag" precisely catches the shots a human actually moves
>2%. With enough data, fit a small logistic model (features: aspect_score, backend,
quad-vs-image-border distance, interior/exterior contrast) to replace the hard
threshold — a genuinely *learned* suspect flag.

### Phase 4 — Regression gate + loop
Wire Phase 1 into a `pytest` marker (`test_detector_no_regression`) that fails if
mean IoU on the log drops. Each new correction enriches the log → re-run → keep the
best. Record deltas in CHANGELOG; keep this file's Evidence table current.

---

## 3. Quick win vs. real fix

- **Quick win (hours):** confidence-aware `expand` (Phase 2.4) + CLAHE-before-detect
  (2.3). Recovers the "cut" class partially with no new detector.
- **Real fix (days):** screen-frame detector + honest multi-hypothesis scoring
  (2.1–2.2), gated by the eval harness. This is what removes the top-edge failure
  class structurally.
- **Heavy option:** fine-tune/replace DocAligner on color-cast projector shots.
  Highest ceiling, but needs a labeled corpus — which the review log is now
  building. Revisit at ≥ a few hundred pairs.

## Status
- [x] Phase 0 — logging live; 11 labeled pairs.
- [x] Phase 1 — eval harness (`research/eval_against_log.py`), faithful to
  production; baseline mean IoU 0.808 / cut 54% recorded above.
- [ ] Phase 2 — detector fixes (next: CLAHE-before-detect probe, then screen-frame detector).
- [ ] Phase 3 — calibration.
- [ ] Phase 4 — regression gate.
