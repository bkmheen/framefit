# framefit — Agent Usage Guide

How another agent (or a human) drives framefit's **semi-automatic review/crop
module**: detect a slide in a photo, propose the 4 corners with a confidence
self-assessment, optionally let a human correct them, crop, and log every decision
as learning data.

> This file is **kept up to date** as the module evolves. When you add/rename a
> flag, change a path, or validate a new recipe, update the matching section and
> the **Maintenance** log at the bottom. Last validated: **2026-07-14, v0.9.0**.

---

## 0. Runtime (read first)

- Run through the **project virtualenv**, not the base interpreter — `cv2`
  (`opencv-python-headless`) and the optional `docaligner` DL backend live there:
  ```bash
  .venv/bin/framefit ...          # console script
  .venv/bin/python -m framefit.cli ...   # equivalent
  ```
  The base/system `python3` typically has **no cv2** and will fail to import.
- Backends: `-b auto` (default; DocAligner → classical fallback), `-b classic`
  (fast, no model), `-b docaligner`.

## 1. Agent vs. human boundary (IMPORTANT)

| Task | Who | Command shape | Blocks? |
|------|-----|---------------|---------|
| **Headless batch crop** (auto-detect → crop → flag uncertain) | **Agent-safe** | no `--review` | No — returns when done |
| **Interactive corner review** (browser, drag 4 points) | **Human only** | `--review` | **Yes — `serve_forever` + opens a browser.** An agent will HANG here. |

**Rule for agents: never run `--review`.** It starts a local web server and blocks
until a human closes it. Do the headless batch instead; it still crops every image
(using the auto corners) and flags the uncertain ones in `framefit_report.tsv` for a
human to refine later. This is the graceful split: agent produces the full first
pass + the uncertain list; a human corrects the flagged corners afterward.

## 2. Validated recipes (all run & verified 2026-07-14)

```bash
# (a) one image → write beside the source as <stem>.jpg, overwriting if present
.venv/bin/framefit "/path/IMG_3634.HEIC" --beside --force

# (b) every HEIC in ONE directory → beside-source .jpg
.venv/bin/framefit "/path/source" --ext heic --beside --force

# (c) AGENT PASS — every HEIC under any source/ folder, recursively,
#     protecting crops a human already hand-corrected in the review log
.venv/bin/framefit "/path/2026-07-13_월" \
    --recurse --under source --ext heic --beside --force --skip-decided

# (d) HUMAN PASS — review only the low-confidence/flagged images, gathered
#     into one queue, auto-accepting the confident ones (opens a browser)
.venv/bin/framefit "/path/2026-07-13_월" \
    --recurse --under source --ext heic --review --only-flagged --beside --force

# (e) headless regeneration of all crops from the decision log (no browser)
.venv/bin/python -m framefit.batch_replay "/path/out_dir"
```

Typical two-phase flow for a folder tree: run **(c)** first (agent or human) so every
image gets a jpg and the uncertain ones are flagged, then run **(d)** (human) to fix
just those.

## 3. Flags reference

| Flag | Effect |
|------|--------|
| `-o DIR` | output directory (default `framefit_out`); ignored when `--beside` |
| `-b {auto,classic,docaligner}` | detection backend (default `auto`) |
| `--beside` | write result next to its source as `<stem>.jpg` (ignores `-o`) |
| `--force` | overwrite an existing output (otherwise that file is skipped) |
| `--ext heic[,jpg]` | when input is a directory, only these extensions |
| `--recurse` | walk subdirectories of a directory input |
| `--under DIRNAME` | with `--recurse`, keep only files under a `DIRNAME/` folder (e.g. `--under source`) |
| `--skip-decided` | skip images already decided in the review log — **protects hand-corrected crops** on a re-run |
| `--review` | **human/browser** interactive corner confirm/edit loop |
| `--only-flagged` | review mode: auto-accept confident images, only stop on flagged ones |
| `--display-max N` | longest side shown in the review page (default 1400) |
| `--review-threshold F` | flag `aspect_score < F` for review (default 0.90) |

## 4. Confidence self-assessment (Module A output)

Each detection carries signals surfaced as a verdict:
- `good` — DocAligner succeeded and aspect score ≥ threshold.
- `suspect` — DL fell back to the classical core (`low_confidence`) **or**
  `aspect_score < review_threshold`. These are the ones `--only-flagged` surfaces.
- `fail` — no quad detected; a human must place 4 corners from scratch.

The batch `framefit_report.tsv` marks `suspect`/`fail` rows as `REVIEW`.

## 5. Learning dataset (for later analysis / model tuning)

Every decision is appended to a **per-host JSONL shard** (sync-conflict-safe across
Macs) and the original + crop are copied content-addressed by SHA-1:

```
~/MarvisHome/Code/framefit/reviews/          # override with $FRAMEFIT_REVIEW_DIR
  log/log-<hostname>.jsonl                    # append-only decisions (read = glob + dedupe on source_sha1)
  originals/<sha1>.jpg                         # downscaled original (cap $FRAMEFIT_ORIG_MAX, default 2560)
  crops/<sha1>.jpg                             # rectified crop
```

Key record fields: `source_sha1` (identity), `backend`, `auto_low_confidence`,
`auto_aspect_score`, `auto_detect_score` (classical composite confidence — a
calibration signal, |r|≈0.65 with accuracy; schema v2), `auto_quad`, `final_quad`,
`action`
(`accept|modify|skip|manual_from_scratch`), `was_modified`, and the calibration
signal **`delta_norm` = max corner move / image diagonal** (how far the human moved
the auto corners, resolution-independent). Cross-tabulate `delta_norm` against the
confidence signals to calibrate the flag / `--review-threshold`.

Read it programmatically:
```python
from framefit import feedback
records = feedback.read_log()      # deduped across all host shards
```

## 6. Public API (for embedding, no CLI)

```python
from framefit import io
from framefit.pipeline import process_image, process_manual   # A (propose), C (crop)
from framefit import feedback                                  # logging + dataset
img = io.load_bgr("IMG.HEIC")
r = process_image(img, backend="auto")     # r.quad, r.backend, r.low_confidence, r.aspect_score
out = process_manual(img, r.quad)          # rectified crop from 4 corners
io.save_bgr(out.image, "IMG.jpg")
```

`review_server.run_review(...)` is the interactive front — **do not call from an
agent** (it blocks on a browser).

---

## Maintenance

Keep this file in sync with the code. Update when any of the below change.

- **Flags**: source of truth is `src/framefit/cli.py::build_parser`.
- **Dataset layout / schema**: `src/framefit/feedback.py` (`review_root`, `build_record`).
- **Interactive server / page**: `src/framefit/review_server.py`.
- **Headless regen**: `src/framefit/batch_replay.py`.

### Change log (agent-facing)
- **2026-07-14 (v0.9.0)** — Multi-hypothesis classical detector (4 candidate cues +
  composite `score_quad`) replaces single-Otsu; fixes color-cast title-cut/ceiling
  grabs. Added `research/eval_against_log.py` (scores detectors vs the human review
  log) and `IMPROVEMENT_PLAN.md`. Auto-path mean IoU 0.808→0.868 on the review set.
- **2026-07-14 (v0.8.0)** — Added the review/crop module: `feedback.py`,
  `review_server.py`, `batch_replay.py`; new CLI flags `--review`, `--only-flagged`,
  `--beside`, `--force`, `--ext`, `--recurse`, `--under`, `--skip-decided`,
  `--display-max`. Review page made viewport-fit (scales to screen, buttons always
  visible). Validated recipes (a)–(e) above on the OpticaImageSensorCongress2026
  photo set (20 HEIC under `source/`; 13 auto-cropped high-confidence, 7 hand-corrected
  crops protected via `--skip-decided`).
