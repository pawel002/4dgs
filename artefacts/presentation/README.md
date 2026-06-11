# Defense Presentation — Precise 3D Reconstruction of Indoor Environments

A reveal.js deck (18 slides, ~20 minutes) with five interactive WebGL demos
built from the thesis content.

## Run it

```bash
./start.sh            # serves on http://localhost:8000 and opens the browser
./start.sh 9000       # custom port
```

The deck **must** be served over HTTP (the demos are ES modules, which browsers
block over `file://`). Everything — reveal.js, three.js, images — is vendored
locally, so **no internet connection is needed** during the defense.

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

## Structure

```
index.html            the deck (18 slides, speaker notes included)
css/theme.css         custom dark theme
assets/               images taken from the thesis (res/, img/)
demos/
  splats.html         hard points vs soft Gaussian splats (same primitives)
  pointcloud.html     COLMAP-style sparse cloud + camera frusta
  trajectory.html     the parametric capture path (ω slider)
  dynamic.html        frozen-topology tracking + emergent static/dynamic split
  velocity.html       linear velocity warm start vs no prediction
  room.js             shared procedural "synthetic room" generator
vendor/               reveal.js 5.1.0 + three.js 0.160 (offline copies)
```

## Placeholders to replace when results are ready

- Slide 16 "Experimental evaluation" — four placeholder cards (PSNR/SSIM tables,
  ablations, real-world capture).
- Slide 17 "Contributions & what's next" — marked as preliminary.
- When real reconstructions exist, a `.splat`/`.ply` viewer (e.g.
  mkkellogg/GaussianSplats3D) can replace the procedural room in
  `demos/splats.html` and `demos/dynamic.html`.
