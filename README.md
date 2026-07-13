# framefit

**Version:** 0.5.0

Detect a document or presentation slide inside a photo, correct its perspective,
and crop it to a clean full-frame image.

Point it at a photo of a projected slide, a whiteboard, or a paper document taken
at an angle — framefit finds the bright rectangle, removes the keystone/perspective
distortion, and saves the flattened, full-frame result.

## Status

MVP: a working package + CLI with two detection backends (classical-CV core and an
optional DocAligner deep-learning backend). See `research/` for the benchmark that
selected the approach.

## Usage

```bash
# a single photo, a list, or whole directories
framefit photo.jpg                       # -> framefit_out/photo_framefit.jpg
framefit slides/ -o out/ -f png          # batch a folder, PNG output
framefit talk.heic -b docaligner --inset 0.01   # DL backend, trim a thin bezel
```

Python API:

```python
import framefit
r = framefit.process_file("photo.jpg", "out.jpg", backend="auto")
print(r.ok, r.aspect_ratio)
```

Pipeline: **load → detect** (on a tone-preprocessed downscale) **→ perspective-warp
& crop from the untouched original → optional bezel inset → save**.

## Repository layout

- `src/framefit/` — the package.
  - `pipeline.py` — high-level `process_image` / `process_file`.
  - `backends/` — `classic.py` (core) and `docaligner.py` (`[dl]` extra) behind a
    common `Detector` interface (`base.py`); `get_backend("auto"|...)`.
  - `preprocess.py`, `geometry.py`, `io.py`, `cli.py`.
- `research/` — the benchmark harness that selected the approach (`RESULTS*.md`,
  `report*.html`).
- `samples/` — local test photos (gitignored, third-party; see policy below).

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
