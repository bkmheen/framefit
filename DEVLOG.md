# Development Log

Chronological engineering notes for framefit. Newest entries at top.

## 2026-07-13 — v0.2.3 — Scenario A: tone preprocessing → C5

User direction: C5 (DL) looked most promising; attack its failure mode by changing
the tone so the dark header/screen boundary is detectable, get coords from the
preprocessed image, but crop from the original. Also keep C2's warp engine and the
idea of feeding C2/C4 results into C5 (scenario B) for later.

Implemented scenario A (chose A1–A3 together):
- A1 gamma/shadow lift (gamma 0.42) — raise dark tones.
- A2 CLAHE on L — local contrast at the screen edge.
- A3 screen-emission isolation — low-threshold lit-region mask (captures the dim
  navy header, excludes the black room) + contrast stretch → near-silhouette.
Harness `run_dl_experiment.py` runs raw/A1/A2/A3 × C5, warps from original, and
scores detected aspect ratio against standard slide ratios (16:9/16:10/4:3/3:2).

## 2026-07-13 — v0.2.2 — Benchmark results & baseline

Ran all 5 candidates on the 11 HEIC samples (see `research/RESULTS.md`,
`research/out/report.html`). Success (found a quad): C1 11/11, C2 11/11, C3 5/11,
C4 11/11, C5 11/11. But "found a quad" != correct — visual review is what matters.

**Key finding.** The dark navy template header blends into the dark auditorium, so
the true top edge is the failure point for every method: C1/C2 clip the header
(content loss), C4 keeps the full slide but leaves keystone uncorrected + black
margins, C5 (DL DocAligner) mislocates the top-left corner and produces a sheared
warp (worst on IMG_3643) at ~231ms/img.

**Baseline / target.** Build on C2's 4-point warp engine but solve the top edge
explicitly: brightness detection tuned to include the header (adaptive/CLAHE),
aspect-ratio-based reconstruction of the weak 4th corner, line-refined edges, C4
full-extent quad as prior, C5 as optional secondary hypothesis scored by
aspect-ratio + coverage.

## 2026-07-13 — v0.2.0 — Candidate survey & benchmark harness

**Open-source survey.** Landscape for "photo → detect slide/document → perspective
correct → crop":
- Classical OpenCV document scanners (andrewdcampbell/OpenCV-Document-Scanner,
  Python-Document-Scanner-OpenCV): grayscale → edge/threshold → largest 4-pt
  contour → `getPerspectiveTransform` + `warpPerspective`. Fast, dependency-light.
- jscanify (puffinsoft): browser/Node, OpenCV.js corner detection + undistort.
- DocAligner (DocsaidLab): DL heatmap regression of 4 corners (facial-keypoint
  style). Robust but needs model download.
- DocScanner (fh2019ustc, IJCV'25): DL document image rectification (handles
  curved warps, heavier).

**Decision.** Benchmark 4 classical candidates in-repo (guaranteed to run offline)
+ evaluate DocAligner as the DL reference. Selected candidates:
- C1 `canny`    — Canny edges → largest convex quad contour.
- C2 `bright`   — Otsu brightness threshold → largest bright quad (tailored to the
  dark-auditorium / bright-screen scene; expected strongest here).
- C3 `hough`    — HoughLinesP → group into H/V edges → 4 intersections.
- C4 `minarearect` — rotated bounding box of largest bright blob (rotation-only
  baseline; no keystone correction, shows the value of true perspective warp).

**Harness.** `research/run_benchmark.py` detects on a 1400px downscale (speed),
warps at full resolution, and emits an HTML grid (overlay + warped output per
sample×candidate) plus results.csv. Outputs gitignored under `research/out/`.

## 2026-07-13 — v0.1.0 — Project bootstrap

**Goal.** Utility that reads a photo, recognizes the document/presentation slide
inside it, corrects perspective, and saves a clean full-frame crop.

**Naming.** Chose `framefit` after checking collisions:
- `fullframe` → collides with camera sensor terminology (bad in an image-sensor domain).
- `autoframe` → collides with video/webcam "auto-framing"; npm name taken.
- `slidefit` → PyPI name already taken; "sliding fit" is a mechanical-tolerance term.
- `framefit` → PyPI free, low GitHub crowding, no strong concept collision. Selected.

**Sample data (local, gitignored).** 11 HEIC photos of projected slides in a dark
auditorium. Two resolutions: 5712×4284 (7 files) and 4032×3024 (4 files). Bright
slide rectangle over dark background, mild keystone + slight rotation.

**Input-spec implications for the code:**
- HEIC input → OpenCV can't decode natively; need `pillow-heif`/`pyheif` preprocessing.
- Dark scene + bright screen → brightness/threshold-based quad detection should work well.
- Weak keystone → 4-point perspective transform is sufficient.
- Two resolutions → keep processing ratio-based, no hardcoded sizes.

**Next.**
1. Open-source survey (document-scanner / dewarp: OpenCV pipelines, jscanify, DocAligner, docTR, etc.).
2. MVP: HEIC load → detect slide quad → perspective-correct → crop → save.
