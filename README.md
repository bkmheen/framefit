# framefit

**Version:** 0.2.0

Detect a document or presentation slide inside a photo, correct its perspective,
and crop it to a clean full-frame image.

Point it at a photo of a projected slide, a whiteboard, or a paper document taken
at an angle — framefit finds the bright rectangle, removes the keystone/perspective
distortion, and saves the flattened, full-frame result.

## Status

Early scaffolding. Benchmarking candidate detection strategies (see `research/`)
to set a baseline before committing to the final pipeline.

## Planned pipeline

1. **Detect** the slide/screen quadrilateral in the image.
2. **Correct** perspective (dewarp the keystone).
3. **Crop** to the slide bounds — full frame, edges trimmed.
4. **Save** the flattened image.

## Repository layout

- `samples/` — local HEIC test photos (gitignored) + `MANIFEST.tsv`.
- `research/` — candidate detection strategies and the benchmark harness.
  - `detectors.py` — classical-CV slide-detection candidates.
  - `run_benchmark.py` — runs every candidate over the samples, writes
    `research/out/report.html` (gitignored) with side-by-side results.

## Project records

- [CHANGELOG.md](CHANGELOG.md) — versioned change history (`major.minor.patch`).
- [DEVLOG.md](DEVLOG.md) — chronological engineering notes and decisions.

If any record file grows too large it will be split by topic (e.g.
`CHANGELOG_<subtitle>.md`) and linked from here.

## License

TBD (intended to be an OSI-approved license before public release).
