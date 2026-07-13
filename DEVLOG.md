# Development Log

Chronological engineering notes for framefit. Newest entries at top.

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
