# Defense Presentation — Precise 3D Reconstruction of Indoor Environments

A reveal.js deck (14 slides, ~12 minutes) with two interactive WebGL demos,
built from the final thesis content (method + measured results).

## Run it

```bash
./start.sh            # serves on http://localhost:8000 and opens the browser
./start.sh 9000       # custom port
```

The deck **must** be served over HTTP (the demos are ES modules, which browsers
block over `file://`). Everything — reveal.js, three.js, MathJax, images — is
vendored locally, so **no internet connection is needed** during the defense.

## Controls

| Key | Action |
|-----|--------|
| `→` / `Space` | next slide |
| `S` | speaker view (notes + timer — notes contain per-slide timing) |
| `F` | fullscreen |
| `Esc` | slide overview |
| `B` | black screen |

The interactive demos respond to mouse drag (orbit) and scroll (zoom) directly
on the slide.

## Structure (~12 min)

| # | Slide | Time |
|---|-------|------|
| 1 | Title | 0:20 |
| 2 | Plan | 0:20 |
| 3 | Motivation | 0:40 |
| 4 | 3D Gaussian Splatting primer (equations) | 1:30 |
| 5 | **Live demo** — real trained 3DGS scene (1M gaussians) | 1:00 |
| 6 | Why per-frame reconstruction of dynamic scenes fails | 1:00 |
| 7 | Proposed 4D pipeline (background / segmentation / hull / tracking) | 1:15 |
| 8 | Shadow-aware hysteresis segmentation (IoU 0.863 → 0.962) | 1:00 |
| 9 | Tracking one splat cloud (5 mechanisms) | 1:30 |
| 10 | **Live demo** — frozen-topology tracking + trajectories | 1:00 |
| 11 | Results: velocity warm start (−84% steps) + iteration budget (7.8 min) | 1:00 |
| 12 | Results: composite quality, leakage, 43× DCT compression | 1:00 |
| 13 | Conclusions & future work | 0:45 |
| 14 | Thank you (Q&A notes) | — |
| 15–16 | Backup slides for Q&A (budget/compression, temporal montage) | — |

```
index.html            the deck (slide order in SLIDE_FILES)
slides/               one file per slide, speaker notes included
css/theme.css         custom light theme
assets/               images from the thesis (res/03_research/results)
demos/
  gsplat.html         real 3DGS rasterizer (antimatter15/splat, WebGL2)
  dynamic.html        frozen-topology tracking + emergent static/dynamic split
  splats.html         hard points vs soft Gaussian splats (spare)
  pointcloud.html     COLMAP-style sparse cloud + camera frusta (spare)
  trajectory.html     parametric capture path (spare)
  room.js             shared procedural "synthetic room" generator
vendor/               reveal.js 5.1.0 + three.js 0.160 + MathJax (offline)
```

Result figures are copied from `artefacts/thesis/res/03_research/results/`;
re-copy them after regenerating thesis figures to keep the deck in sync.
