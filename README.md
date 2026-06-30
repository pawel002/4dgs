# Temporal (4D) Gaussian Splatting of a Dynamic Room

A master's-thesis project on reconstructing a **dynamic indoor scene** — a subject moving
through an otherwise static, furnished room — as a **time-varying 3D Gaussian-Splatting
field** that can be rendered from any viewpoint at any instant.

The central idea is a **static / dynamic decomposition**: the rigid room is reconstructed
once, the moving subject is reconstructed as *one Gaussian cloud per video frame*, and the
two are recombined at render time. Keeping the primitive count and appearance fixed after the
first frame turns per-frame reconstruction into cheap **tracking**, and makes the result
temporally coherent and highly compressible.

## Pipeline (high level)

```
 Blender ─► segmentation ─►  ┌─ static background ─► splatfacto ──────┐
 (synthetic   (background     │                                       ├─► merge renderer ─► play
  capture +    subtraction,   └─ dynamic foreground ─► temporal-      │   (viser, composite)
  exact poses) no model)         (per-frame clouds)    splatfacto ─►──┘            │
                                                                                   └─► DCT compression
```

1. **Synthetic capture (Blender).** A furnished room is rendered with a moving subject, seen
   by a rig of fixed cameras (multi-view parallax) plus a roving camera for the background.
   Camera poses and intrinsics are **exported exactly** from the scene, so no
   Structure-from-Motion / pose estimation is needed.
2. **Foreground segmentation (no model).** The subject is isolated onto a black background by
   **background subtraction** against an empty-room plate — a pure pixelwise difference, with
   no learned segmentation network.
3. **Static background.** The rigid room is reconstructed once with stock `splatfacto`.
4. **Dynamic foreground.** `temporal-splatfacto` seeds Gaussians from the subject's **visual
   hull**, densely fits frame 0, then **tracks** that one cloud through every later frame
   under a fixed count and frozen colour (velocity prediction + per-frame hull snapping keep
   it on the subject).
5. **Merge & play.** A `viser` renderer composites the static background with the per-frame
   foreground clouds and gives playback controls.
6. **Compression.** Because every frame shares the same primitives, each attribute is a smooth
   time series and the whole sequence is compressed in a temporal-frequency (DCT) basis
   (~10× at ~30 dB).

## Repository layout

| Path | What it is |
|---|---|
| `nerfstudio/` | **Core implementation.** The custom `temporal-splatfacto` method, the merge renderer, and the temporal compressor, built on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio). See **`nerfstudio/scripts.md`** for end-to-end usage. |
| `room-blender/` | Blender scene and `bpy` scripts that render the synthetic room and export exact poses. |
| `output/` | Rendered datasets — one folder per fixed camera (`static1`…`static4`) plus `dynamic1`. |
| `4dgs/` | The original 4D Gaussian-Splatting reference code (`train_video.py`) that `temporal-splatfacto` is a nerfstudio port of. |
| `colmap/` | Optional COLMAP Structure-from-Motion pose-recovery pipeline (superseded by direct Blender pose export, kept for comparison). |
| `artefacts/thesis/` | The LaTeX master's thesis (full method, derivations, and results). |
| `artefacts/presentation/` | Reveal.js slide deck and demos. |

## Where to start

- **Run the reconstruction:** `nerfstudio/scripts.md` (training, viewing, export, the merge
  viewer, and compression — all the commands and flags).
- **Understand the method:** `artefacts/thesis/tex/03_research.tex` (the research chapter).
