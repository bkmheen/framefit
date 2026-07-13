# Candidate benchmark results (baseline)

Dataset: 11 HEIC photos of projected slides (dark auditorium, DynamicOptics
template with a **dark navy header band**). Detection on 1400px downscale; warp at
full resolution. Regenerate with `python research/run_benchmark.py`; visual grid at
`research/out/report.html`.

## Quantitative summary

| candidate | approach | success | avg area% | avg ms |
|-----------|----------|:---:|:---:|:---:|
| C1 `canny` | Canny edges → largest quad contour | 11/11 | 40.4 | 4 |
| C2 `bright` | Otsu brightness → largest bright quad | 11/11 | 41.1 | 3 |
| C3 `hough` | HoughLinesP → 4-edge intersection | 5/11 | 39.9 | 14 |
| C4 `minarearect` | rotated bbox of largest bright blob | 11/11 | 47.4 | 3 |
| C5 `docaligner` | DL heatmap corner regression (ONNX) | 11/11 | 44.7 | 231 |

"success" = a quad was returned, **not** that it is correct. Quality was judged
visually (below). area% ~40 for C1/C2 vs ~47 for C4/C5 reflects C1/C2 cropping
*inside* the true slide (clipping the header).

## Qualitative findings (the important part)

**The decisive difficulty is the top edge / dark header.** The template's navy
header bar has brightness close to the dark auditorium background, so the true top
boundary is weak. Every method struggles there:

- **C1 canny** — clean perspective crop, but **clips the dark header** (title row
  "Lasercom: Ground to Ground" + logo lost). Content loss → unacceptable as-is.
- **C2 bright** — cleanest keystone correction of all, but **clips the header** for
  the same reason (Otsu puts the navy band below threshold). Best *warp engine* if
  the top-edge problem is solved.
- **C3 hough** — fails on 6/11 (needs 2 clear H + 2 clear V lines; the weak
  top/side edges in the dark break it). Not robust here.
- **C4 minarearect** — **most robust at capturing the full slide** including the
  header, but (a) includes black wedge margins and (b) applies **no keystone
  correction** (output stays trapezoidal). Good extent, wrong geometry.
- **C5 docaligner (DL)** — trained on paper documents; on these dark-header slides
  it **mislocates the top-left corner** (drags it down), producing a clipped and
  **sheared** result (see IMG_3643). Also 50–75× slower (231 ms vs ~3 ms). Not a
  turnkey win here.

## Baseline conclusion → development target

No off-the-shelf candidate is sufficient on its own. The two useful signals:
- **C2** gives the correct *perspective-correction engine* (4-point warp) and the
  cleanest content when it does not clip.
- **C4** gives the correct *full extent* (captures the dark header) but no geometry
  correction.

**Proposed framefit approach** = solve the dark-header top edge explicitly, then
warp:
1. Brightness/contrast detection tuned to include the navy header band (lower/
   adaptive threshold, or CLAHE before Otsu), OR detect the projector-screen
   bright frame rather than slide content.
2. Use the **known slide aspect ratio** (16:9 / 16:10 / 4:3) plus the 3 reliable
   corners to reconstruct/verify the weak 4th (top) corner.
3. Refine the 4 edges with line fitting for a true keystone-corrected quad
   (C2-style warp), using C4's full-extent quad as a fallback/prior.
4. Optionally keep C5 (DL) as a secondary hypothesis and pick the best by an
   aspect-ratio + content-coverage score.

This benchmark and `report.html` are the reference bar for the pipeline we build.
