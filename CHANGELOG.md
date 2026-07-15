# Changelog

All notable changes to framefit are recorded here.
Versioning: `major.minor.patch` (initial major = 0).

## [Unreleased]

### Fixed
- **`--review` can now re-open an already-decided image for correction.** Review
  mode always auto-skipped any image with a prior decision in the review log, with
  no way to override it — a human asking to re-adjust a single already-decided file
  got a "완료 (이전 결정 건너뜀)" screen instead of the editor. The CLI now threads
  `--skip-decided` into review mode (default off), and `run_review()` takes a
  `skip_decided` parameter. A bare `framefit "<img>" --review` re-presents the image
  for editing; pass `--skip-decided` to keep the resume-safety behavior that protects
  earlier hand-corrections.

## [0.9.1] - 2026-07-15

### Changed
- **Beside-the-source is now the DEFAULT output rule.** A bare `framefit <imgs>`
  (no `-o`, no `--beside`) writes each result next to its source with the **same
  filename and a `.jpg` extension** — the standing convention: same name, extension
  only changed, no arbitrary output directory. This applies uniformly whether a
  human or another agent invokes framefit. Pass `-o DIR` to opt out and collect
  crops elsewhere as `<stem>_framefit.<fmt>`. The `--beside` flag is now redundant
  but still accepted for explicitness/back-compat. (`AGENTS.md` updated to match.)

### Fixed
- **`--beside` no longer creates a stray `framefit_out/`.** Previously, even with
  `--beside` (write each crop next to its source), the batch QA report was still
  written to the default `framefit_out/` directory, forcing an unwanted directory
  into existence in the working dir. The report now lands in the common parent
  directory of the crops (i.e. beside the sources), so `--beside` creates no
  arbitrary output directory of its own.

## [0.9.0] - 2026-07-14

### Changed
- **Multi-hypothesis classical detector.** `backends/classic.py` no longer trusts a
  single global-Otsu bright blob. It now proposes candidate quads from four cues
  (bright-Otsu, hole-filled Otsu, Canny screen boundary, HSV-value) and picks the
  best with a composite `score_quad` — edge support, interior/surround contrast,
  exterior-quiet (penalizes a quad that cuts *across* the slide), aspect, area,
  border-touch penalty. Fixes the dominant color-cast failure where the detector
  grabbed the bright inner image and cut the title (or swallowed ceiling lights).
  Measured on the review log (auto path, n=11): mean IoU 0.808 → **0.868**, worst
  shot 0.239 → **0.894**, cut-rate 54% → 46%, with zero regressions on the
  previously-correct shots. See `IMPROVEMENT_PLAN.md`.

### Added
- **Detection confidence signal (`auto_detect_score`, schema v2).** The winning
  candidate's composite `score_quad` is now surfaced on `Result.detect_score` and
  logged with every review decision. On the review log it correlates with accuracy
  (|r|≈0.65) — a better calibration signal than `aspect_score`. The review flag is
  deliberately *not* re-thresholded yet (n too small); the signal is accumulated
  for a learned threshold once the dataset grows (Phase 3 of `IMPROVEMENT_PLAN.md`).
- **Offline eval harness** `research/eval_against_log.py` — scores any detector
  against the human `final_quad` labels via the production `process_image` path
  (IoU / delta_norm / top-edge error); works from the SHA-1 dataset copies even
  after source photos are deleted. The gate for every future detector change.
- **`IMPROVEMENT_PLAN.md`** — data-driven, staged detection-improvement plan
  (evidence → harness → detector fixes → calibration → regression gate).

## [0.8.0] - 2026-07-13

### Added
- **Reliable low-confidence flag (B).** `Result.low_confidence` is set when the
  auto backend's DL detector fails and it falls back to the classical core.
  Verified this cleanly separates the good shots (DocAligner OK on navy + white
  templates) from the hard bright pull-down-screen shots (DocAligner FAIL) — it
  catches the cut cases the aspect-score threshold missed (e.g. IMG_3658). The CLI
  review flag and `framefit_report.tsv` now use it (with score as a secondary
  signal) and record the actual backend used.
- **Manual corner interface (C).**
  - `framefit --pick` writes a self-contained HTML corner-picker per image; click
    the 4 corners in a browser and it prints the ready-to-run command (headless —
    just writes a file). `picker.py`.
  - `framefit --corners "x1,y1 x2,y2 x3,y3 x4,y4"` and `pipeline.process_manual()`
    rectify from user-supplied corners, bypassing detection.

## [0.7.0] - 2026-07-13

### Added
- **Safety expansion** (`process_image(expand=0.04)`, default on; CLI `--expand`).
  Grows the detected quad outward before warping so a slightly-inaccurate detection
  never crops into content (fixes titles flush to the slide's top edge, e.g.
  IMG_3636/3637). The refine pass reclaims the added margin where it is genuinely
  empty, so the common case stays tight (navy set AR essentially unchanged).
- **QA report** (`framefit_report.tsv` written to the output dir; CLI
  `--review-threshold`, default 0.90). Lists per-file aspect ratio + score and
  flags low-score / missed / errored results as REVIEW so hard cases (e.g. the
  bright pull-down-screen template) surface for manual check. The run summary
  prints how many were flagged.

### Fixed
- `inset_quad` ignored negative fractions (expansion was a silent no-op); it now
  scales about the centroid in both directions.

## [0.6.2] - 2026-07-13

### Added
- **Header-aware top trim** (`trim_dark_margins(header_aware=True)`, default on).
  Some shots leave a dim, non-black gap above the slide (~0.16, a warm/neutral room
  surround) that the near-black rule can't reach (IMG_3640/3641/3642). Measured
  discriminator: the gap is "blueness"-negative while the navy header flips strongly
  positive — so the top is trimmed down to the first colored-header row. Safely
  gated: bails on bright content, when no colored header is present (other
  templates), or when the region above isn't dark margin — so it fixes navy-header
  slides and is inert elsewhere. Never cuts the header (stops at its top edge).

## [0.6.1] - 2026-07-13

### Fixed
- **Refine no longer cuts content.** `trim_dark_margins` default `dark_ratio`
  lowered 0.40 → 0.12. Root cause (measured): the dark navy slide header sits at
  ~0.16 brightness while the empty room/bezel above it is ~0.09; the old 0.40
  threshold treated the header as margin and cut it (e.g. the "DYNAMIC OPTICS"
  logo top on IMG_3646). The lower near-black threshold trims only the empty gap
  and stops at the header. Never cuts on the sample set; most tops ≤3% residual,
  a few ~10% (safe margin retained over cutting).
- Added a regression test for preserving a dim (non-black) header band.

### Investigated (not adopted)
- 180° re-detection idea: measured on all 11 — the top margin is essentially
  unchanged (bottom fits well due to bright-content contrast, not position; the
  low-contrast dark-header edge overshoots regardless of orientation). Rejected.

## [0.6.0] - 2026-07-13

### Added
- **Edge refinement** (`geometry.trim_dark_margins`): after warping, uniformly-dark
  border bands left by an imprecise edge (typically the top, above a dark header)
  are trimmed so the slide fills the frame. Guarded by a variance test (textured
  content is never cut) and a per-edge trim cap. On by default in `process_image` /
  `process_file`; `--no-refine` disables it.
- Chosen over the alternatives after measuring on the samples: re-detection closed
  only ~half the gap and failed on 4/11; aspect-snap was partial; trim reached 0%
  top margin on all 11 with content intact. Reported aspect ratio now reflects the
  final (trimmed) image.

## [0.5.0] - 2026-07-13

### Added
- `AutoDetector` best-of/fallback backend: prefers DocAligner (when installed),
  falls back to the classical core, and picks the more slide-like quad by
  aspect-ratio score. Instantiated once, reused across a batch (model loads once).
- Smoke tests (`tests/`, `dev` extra) covering geometry, classic backend, and the
  process pipeline — no DL/HEIC needed.
- `research/make_gallery.py`: builds a before/after gallery of the installed
  package over the samples (`research/out_final/`, gitignored).

### Changed
- `get_backend("auto")` now returns `AutoDetector` (was: bare DocAligner/classic).

## [0.4.1] - 2026-07-13

### Changed
- Set copyright holder / author to **Bongki Mheen (bkmheen@gmail.com)** in NOTICE
  and pyproject; aligned git commit identity for future commits.

## [0.4.0] - 2026-07-13

### Added
- **MVP package `src/framefit/`** with a working CLI (`framefit`) and Python API.
  - `backends/`: `Detector` interface + `ClassicDetector` (permissive core: OpenCV
    brightness quad + minAreaRect fallback) and `DocAlignerDetector` (`[dl]` extra:
    A1 gamma-lift → DocAligner corners). `get_backend("auto")` prefers DL when
    installed, else classic.
  - `pipeline.py`: load → detect on a preprocessed downscale → warp/crop from the
    untouched original → optional bezel inset. Returns a `Result` with aspect
    ratio + score.
  - `io.py` (HEIC via optional `heic` extra), `preprocess.py`, `geometry.py`,
    `cli.py`.
- `framefit` console entry point in pyproject.

## [0.3.0] - 2026-07-13

### Added
- **License: Apache-2.0.** `LICENSE` (canonical text) + `NOTICE` (framefit
  copyright + third-party breakdown).
- `pyproject.toml` establishing the license-clean separation:
  permissive core (numpy / opencv-python-headless / Pillow) + opt-in extras
  `[dl]` (docaligner-docsaid) and `[heic]` (pillow-heif), plus `[all]`.
- README: Installation (extras), License, third-party table, and a Test-assets
  policy (samples/ are third-party copyrighted; never published).

### Notes
- Licensing findings that drove this: DocAligner code is Apache-2.0 but its model
  weights are unspecified (downloaded at runtime, not redistributed here);
  pillow-heif wheels bundle x265 (GPLv2) and HEIC/HEVC is patent-encumbered — both
  isolated behind opt-in extras so the core stays permissive/patent-clean.
- Package source (`src/framefit/`) is forthcoming; pyproject declares the intended
  structure ahead of the implementation.

## [0.2.4] - 2026-07-13

### Findings
- Scenario A result (`research/RESULTS_A.md`): **A1 gamma/shadow lift → C5 is the
  clear winner.** Lifting shadows reveals the whole projector screen, so DocAligner
  locks the true 4 corners. Fixes both the header-clipping (IMG_3640) and the shear
  (IMG_3643) seen with raw C5. Mean aspect-ratio score 0.99 (raw 0.95); no
  regressions. A2/A3 did not beat raw.
- Adopted MVP core: A1 gamma-lift → C5 DocAligner → warp-from-original.

## [0.2.3] - 2026-07-13

### Added
- Scenario A experiment: tone preprocessing → C5 (DocAligner).
  - `preprocess.py`: A1 gamma/shadow lift, A2 CLAHE, A3 screen-emission isolation.
  - `run_dl_experiment.py`: runs raw/A1/A2/A3 × C5 over the samples, warps from
    the untouched original, scores detected aspect ratio vs standard slide ratios,
    emits `research/out_dl/report_dl.html` + `results_dl.csv`.
- Detection image and output image are separated: preprocessing only feeds corner
  detection; crop/warp always uses the original.

## [0.2.2] - 2026-07-13

### Added
- `research/RESULTS.md`: baseline benchmark of all 5 candidates on the 11-sample
  set (quantitative table + qualitative findings + development target).

### Findings
- Decisive difficulty = the dark navy header edge blending into the dark room.
  C1/C2 clip it; C4 keeps full extent but no keystone correction; C5 (DL)
  mislocates the top-left corner (sheared output) and is ~231ms/img.
- Development direction: brightness detection tuned to include the header +
  aspect-ratio-based 4th-corner reconstruction + line-refined keystone warp
  (C2 engine), with C4 as extent prior and C5 as secondary hypothesis.

## [0.2.1] - 2026-07-13

### Fixed
- C3 Hough detector crashed on OpenCV 5.0 (`HoughLinesP` returns `(N,4)`);
  reshape made it robust. Now 5/11 on samples.

### Added
- C5 deep-learning candidate `detectors_dl.py` (DocAligner heatmap corner model);
  `run_benchmark.py` folds it in when available (`FRAMEFIT_DL=0` to skip).

## [0.2.0] - 2026-07-13

### Added
- `research/` benchmark harness for comparing slide-detection strategies:
  - `detectors.py`: 4 classical-CV candidates — C1 Canny+quad contour,
    C2 Otsu brightness+quad (scene-tailored), C3 Hough 4-edge intersection,
    C4 minAreaRect rotation baseline.
  - `run_benchmark.py`: loads HEIC samples, runs each candidate, applies
    4-point perspective warp, and emits `research/out/report.html` +
    `results.csv` (both gitignored).
- Open-source survey recorded in DEVLOG (OpenCV scanners, jscanify,
  DocAligner, DocScanner).

### Notes
- DL candidate (DocAligner) evaluated separately; requires model download.

## [0.1.0] - 2026-07-13

### Added
- Initial project scaffold: `README.md`, `.gitignore` (Python/OpenCV oriented).
- Development records: `CHANGELOG.md`, `DEVLOG.md`.
- Local sample set under `samples/` (gitignored): 11 HEIC photos of projected
  conference slides collected from the Optica Image Sensor Congress 2026 Plenary2
  source folders, with `samples/MANIFEST.tsv` mapping each file to its source subdir.

### Notes
- No source code yet. Next: open-source survey, then MVP pipeline
  (HEIC load → slide detection → perspective correction → crop → save).
