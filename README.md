# framefit

**Version:** 0.3.0

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

## Installation (planned layout)

framefit is split so that license/patent exposure is **opt-in**:

```bash
pip install framefit          # permissive core: classical-CV detectors + warp engine
pip install "framefit[dl]"    # + DocAligner deep-learning corner detector
pip install "framefit[heic]"  # + HEIC/HEIF input support
pip install "framefit[all]"   # everything
```

The core has no GPL dependency, no patent-encumbered codec, and no
ambiguously-licensed model weights. The `dl` and `heic` extras add capability that
carries its own licensing implications (see below) — you choose whether to accept
them.

## License

framefit is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) and
[NOTICE](NOTICE).

### Third-party components & the extras

framefit **depends on** the following but does **not** redistribute them (they are
installed separately by pip, each under its own license):

| Component | License | Where |
|-----------|---------|-------|
| OpenCV (`opencv-python-headless`) | Apache-2.0 | core |
| NumPy | BSD-3-Clause | core |
| Pillow | MIT-CMU (HPND) | core |
| DocAligner (`docaligner-docsaid`, `capybara`) | Apache-2.0 | `[dl]` extra |
| pillow-heif | BSD-3-Clause* | `[heic]` extra |

Notes you should be aware of:
- **DocAligner model weights** are downloaded at runtime from the DocAligner
  project and governed by *its* terms, not framefit's. framefit never ships the
  weights. If you need commercial-use certainty, prefer the classical-CV core
  backend or confirm terms with the DocAligner authors.
- **pillow-heif** binary wheels bundle native codecs including **x265 (GPLv2)**, so
  the distributed wheel is effectively GPLv2, and **HEIC/HEVC is
  patent-encumbered**. Installing the `heic` extra is your choice.

*See [NOTICE](NOTICE) for the full third-party breakdown.*

## Test assets policy

The photos under `samples/` are **third-party copyrighted material** (conference
presentation slides, including third-party logos) and are **gitignored — never
published** in this repository. Public releases must use only test images we have
the right to distribute (self-captured, synthetic, or explicitly-licensed).
