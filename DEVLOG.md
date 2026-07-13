# Development Log

Chronological engineering notes for framefit. Newest entries at top.

## 2026-07-13 — v0.6.1 — Fix refine over-crop; reject 180° idea

User found the 0.6.0 trim cut content (IMG_3646 "DYNAMIC OPTICS" logo top) — a
regression, since pre-trim never cut. Two threads:

1. **User's 180° idea** (rotate so the well-fit bottom handles the top, then rotate
   back). Tested on all 11: top empty margin ~unchanged (normal vs rot180 both
   ~11–12%), one case worse. Reason: the bottom fits well because of bright-content
   *contrast*, not position; rotating carries the low-contrast dark-header edge to
   the bottom but it still overshoots. Rejected (kept as a documented negative).

2. **Real fix.** Measured the top-region profile of a pre-trim output: empty gap is
   near-black (~0.09) with neutral/negative blueness; the navy header is ~0.16+ and
   blue-positive. The old `dark_ratio=0.40` swept the header into "margin" and cut
   it. Lowered to 0.12 (near-black only) → trims the empty gap, stops at the header.
   Verified: IMG_3646 logo intact, IMG_3640 fine; most tops ≤3%, a few ~10% (safe).
   Priority honored: never cut > minimize margin. Regression test added.

## 2026-07-13 — v0.6.0 — Edge refinement (top-margin trim)

User noticed a residual dark band at the top of the outputs. Measured it: ~15%
dark margin on the TOP edge only (bottom/left/right ~0%) — DocAligner places the
top edge slightly high (lowest-contrast edge). Compared three fixes on the 11
outputs: (a) user's re-detection idea — directionally right (AR converged to
16:10) but failed on 4/11 (slide fills frame → no corners) and only halved the gap
+ double-warp; (b) aspect-snap — partial/inconsistent; (c) projection-profile trim
— 0% top margin on all 11, content intact, ~1ms, no deps. Picked (c).

Implemented `trim_dark_margins`: per-edge, trims border rows/cols that are both
dark AND low-variance (so textured content survives), capped per edge. Wired into
the pipeline as `refine=True` (default), `--no-refine` to disable; reported AR now
comes from the final image. Tests cover trim + the textured-edge guard.

## 2026-07-13 — v0.5.0 — Quality pass: auto best-of + tests + gallery

Robustness polish. `AutoDetector` runs the available backends and keeps the more
slide-like quad (aspect-ratio score), with DocAligner preferred and classic as a
guaranteed offline fallback. It's a single Detector instance (model loads once),
so batch CLI stays fast. Added smoke tests for the permissive core and a gallery
generator to eyeball the final results across all 11 samples. (v0.4.1 set the
copyright/author to Bongki Mheen — see [[framefit-copyright]].) Next: run tests +
gallery, confirm quality, then public-release prep.

## 2026-07-13 — v0.4.0 — MVP package + CLI

Turned the validated research into `src/framefit/`. Backend abstraction realises
the license separation in code: the permissive `classic` backend (OpenCV only) is
the default core; the `docaligner` backend lives behind the `[dl]` extra and pairs
the A1 gamma-lift preprocess with the DL corner model (the benchmark winner). HEIC
loading sits behind the `[heic]` extra with a helpful error if missing.

Pipeline mirrors the research harness exactly: detect on a 1400px preprocessed
downscale, map corners back, warp from the untouched original, optional inset for
the screen bezel. CLI supports files/dirs, backend choice, format, inset. Next:
install -e and verify on the samples.

## 2026-07-13 — v0.3.0 — Licensing decided: Apache-2.0 + core/extras split

Investigated licensing for public release. Code licenses of the whole stack are
permissive (OpenCV/DocAligner/capybara Apache-2.0, pillow-heif BSD-3, onnxruntime
MIT, NumPy BSD, Pillow MIT-CMU). Two real risks surfaced:
1. **DocAligner model weights**: code is Apache-2.0 but the weights' license is
   unspecified; trained on SmartDoc2015 / MIDV-500·2019·2020 / CORD / self-collected
   online docs. We never redistribute the weights (runtime download), so the risk
   stays upstream.
2. **HEIC**: pillow-heif wheels bundle x265 (GPLv2) → effectively GPLv2, and
   HEIC/HEVC is patent-encumbered.

Decision: **framefit code = Apache-2.0** (explicit patent grant fits the HEVC
domain; consistent with our Apache deps; clean contribution terms). Separation of
"development vs published" implemented the standard way — depend-don't-vendor +
permissive core with opt-in extras (`[dl]`, `[heic]`). NOTICE burden is ~zero now
(no vendored third-party source; deps ship no NOTICE files); only relevant if we
later bundle into a single binary/image. samples/ are third-party copyrighted and
stay gitignored; public releases need our own/synthetic test images.

Added: LICENSE, NOTICE, pyproject.toml (core + extras), README licensing sections.

## 2026-07-13 — v0.2.4 — Scenario A result: A1 gamma-lift → C5 wins

Ran raw/A1/A2/A3 × C5 on all 11. **A1 (gamma/shadow lift) is the clear winner**:
mean aspect-ratio score 0.99 vs raw 0.95, detected area up (~0.37→0.5), and — the
point — visually it recovers the full slide including the dark header and removes
the shear that broke raw C5 on IMG_3643. A2/A3 no better than raw on this set.

Mechanism: lifting shadows makes the whole physical projector screen visible, so
the DL model sees a clean rectangle and locks the true corners; we then warp from
the untouched original. Confirms the user's hypothesis exactly.

MVP core decided: **A1 gamma-lift → C5 DocAligner → warp-from-original.** Residual:
thin screen-bezel margin (benign). Next: tighten bezel; scenario B1 (C2/C4 ROI
prior → C5) as fallback; scenario-C scoring for auto-pick. See `research/RESULTS_A.md`.

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
