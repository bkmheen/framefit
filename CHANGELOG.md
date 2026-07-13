# Changelog

All notable changes to framefit are recorded here.
Versioning: `major.minor.patch` (initial major = 0).

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
