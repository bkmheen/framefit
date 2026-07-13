# Changelog

All notable changes to framefit are recorded here.
Versioning: `major.minor.patch` (initial major = 0).

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
