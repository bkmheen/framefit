# Scenario A results — tone preprocessing → C5 (DocAligner)

Idea (user): the DL corner detector was best in principle but mislocated the
top corners because the dark navy header blends into the dark room. Fix the *tone*
so the screen boundary is visible, take coordinates from the preprocessed image,
and crop/warp from the **untouched original**.

Run: `python research/run_dl_experiment.py` → `research/out_dl/report_dl.html`.

## Aspect-ratio score (mean over 11 samples; 1.0 = matches a standard slide AR)

| variant | ok | mean AR score | note |
|---------|:---:|:---:|------|
| raw (no preprocess) | 11/11 | 0.95 | often clips header → AR too wide (~1.9) |
| **A1 gamma/shadow lift** | 11/11 | **0.99** | **best** — AR normalizes to ~1.6 (16:10) |
| A2 CLAHE | 11/11 | 0.95 | mild help, less consistent |
| A3 screen-isolation | 11/11 | 0.94 | no better than raw here |

Detected area also rises with A1 (~0.37→0.47–0.53), i.e. it now captures the full
slide instead of a header-clipped subregion.

## Visual verdict

**A1 (gamma/shadow lift) → C5 is the clear winner.** Lifting the shadows reveals
the whole physical projector screen, so DocAligner locks onto the true 4 corners.

- IMG_3640: raw clipped the "Lasercom: Ground to Ground" title; **A1 recovers the
  full slide** (both logos + header), keystone-corrected, no clipping.
- IMG_3643: raw produced a **sheared** table; **A1 is square and complete.**
- No regressions on the easy samples (A1 AR score 0.98–1.00 across all 11).

Residual: A1 tends to include a thin screen-bezel margin (benign — no content loss;
can be tightened later with a small inward inset or edge refine).

## Decision

Adopt **A1 gamma-lift → C5 DocAligner → warp-from-original** as the framefit MVP
core. Next options:
- Tighten the bezel margin (inward inset / edge-snap).
- Scenario B1 (C2/C4 ROI prior → C5) as robustness fallback for hard shots.
- Add scenario-C scoring (aspect + coverage) to auto-pick when methods disagree.
