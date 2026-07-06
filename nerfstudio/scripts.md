# Gaussian Splatting reconstruction — `dynamic1` room

Commands used to reconstruct the room in `../output/dynamic1` (150 images +
`transforms.json`) as a 3D Gaussian Splatting model with nerfstudio's
`splatfacto`, and to view it live in the viser viewer.

All commands are run from the `nerfstudio/` directory using the local
virtualenv at `.venv`. GPU is pinned with `CUDA_VISIBLE_DEVICES=0`.

---

## 0. Environment notes / one-time fixes

This machine needed two fixes before training would run:

### a) `transforms.json` had no explicit camera intrinsics
The input `transforms.json` used the Blender/instant-ngp convention
(`camera_angle_x`, `w`, `h`) but nerfstudio's dataparser requires explicit
focal lengths (`fl_x`, `fl_y`, `cx`, `cy`). We computed them from the field of
view and added them globally (original saved as `transforms.json.orig`):

```
fl_x = fl_y = 0.5 * w / tan(0.5 * camera_angle_x)   # = 1800 px for w=1920
cx = w/2 = 960 ,  cy = h/2 = 540
```

**Permanent fix:** the Blender exporters at `../room-blender/script-static.py`
and `../room-blender/script-dynamic.py` now emit `fl_x`, `fl_y`, `cx`, `cy`
(and `camera_angle_y`) directly, so freshly re-rendered datasets load into
nerfstudio without this manual patch.

### IMPORTANT — `output/dynamic1` is a single fixed viewpoint
All 150 frames in `output/dynamic1/transforms.json` share **one identical**
`transform_matrix`. `script-dynamic.py` reads `cam.matrix_world` every frame
(correct), but the `dynamic1` camera in the `.blend` has **no keyframed
motion**, so every pose is the same point. Gaussian Splatting (static *or*
dynamic) cannot recover 3D geometry from a single fixed view — there is no
parallax. The "random cube of colored points" result is the expected failure
mode. To get a real reconstruction you must either:
  - animate the `dynamic1` camera in Blender (keyframe it moving through the
    room) and re-render, **or**
  - use `script-static.py`, which has 4 distinct fixed cameras
    (`static1`..`static4`) — merge their views into one dataset for parallax.

### b) The GPU (RTX 6000 Ada, sm_89) is newer than the default CUDA toolkit
`gsplat` JIT-compiles a CUDA extension on first run. The system `nvcc`
(`/usr/bin/nvcc`) is CUDA 11.5 and rejects `compute_89`. A CUDA 12.4 toolkit
is installed at `/usr/local/cuda-12.4`, which does support sm_89, so we point
the build at it with `CUDA_HOME` + `PATH`:

```
CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH
```

There was also a minor Pillow-version incompatibility in
`nerfstudio/data/utils/data_utils.py` (`pil_to_numpy`); it now falls back to
`np.asarray(im)` when PIL's fast raw encoder API is unavailable.

---

## Camera-bounding-box random initialization (custom)

When there is no SfM/point-cloud seed, splatfacto now initializes the random
Gaussian cloud **inside the bounding box of all training cameras, expanded 2×**,
instead of a fixed `random_scale=10` cube. This places the initial Gaussians
where the cameras actually look, which converges much better for room scenes.

Implemented in:
- `nerfstudio/models/splatfacto.py` — `SplatfactoModel._random_init_means()`
  plus config fields `random_init_from_camera_bbox` (default `True`) and
  `random_init_box_scale` (default `2.0`).
- `nerfstudio/pipelines/base_pipeline.py` — passes training-camera positions to
  the model via `metadata["camera_positions"]`.

Behavior: box size = `(camera_bbox_max - camera_bbox_min) * random_init_box_scale`,
centered on the cameras (in the dataparser's normalized frame). If camera
positions are unavailable or all cameras are coincident (degenerate bbox, e.g.
the broken single-view `dynamic1`), it falls back to the `random_scale` cube.

Tune from the CLI:
```bash
--pipeline.model.random_init_box_scale 2.0          # box = 2x camera bbox
--pipeline.model.random_init_from_camera_bbox False # disable, use random_scale cube
```

## 1. Train — `ns-train splatfacto`

Runs the Gaussian Splatting optimization (30 000 iterations by default) and
launches the **viser** web viewer so the reconstruction can be watched live.

```bash
CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/ns-train splatfacto \
  --data ../output/dynamic1 \
  --output-dir splats/dynamic \
  --vis viewer \
  --viewer.quit-on-train-completion False \
  --viewer.websocket-port 7007 \
  --experiment-name dynamic \
  nerfstudio-data
```

What the flags do:
- `splatfacto` — nerfstudio's 3D Gaussian Splatting method.
- `--data` — folder with `transforms.json` + `images/`.
- `--output-dir splats/dynamic` — where checkpoints/config are written
  (`splats/dynamic/dynamic/splatfacto/<timestamp>/`).
- `--vis viewer` — enable the live viser viewer (no eval images logged).
- `--viewer.quit-on-train-completion False` — keep the viewer running after
  training so you can keep inspecting the result.
- `--viewer.websocket-port 7007` — viewer port.
- `--experiment-name dynamic` — names the run folder `dynamic`.
- `nerfstudio-data` — the dataparser (reads `transforms.json`). No COLMAP
  point cloud is present, so splatfacto uses random point-cloud init.

**Open the viewer at:** http://localhost:7007
(If working over SSH, forward the port: `ssh -L 7007:localhost:7007 <host>`.)

---

## 2. View an already-trained model — `ns-viewer` (optional)

Re-open the viser viewer for a finished run without retraining:

```bash
CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/ns-viewer \
  --load-config splats/dynamic/dynamic/splatfacto/<timestamp>/config.yml
```

---

## 3. Export the splat — `ns-export gaussian-splat`

Writes the trained Gaussians to a `.ply` (loadable in any 3DGS viewer).

```bash
CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/ns-export gaussian-splat \
  --load-config splats/dynamic/dynamic/splatfacto/<timestamp>/config.yml \
  --output-dir splats/dynamic/export
```

Produces `splats/dynamic/export/splat.ply`.

---

# Temporal (4D) Gaussian Splatting — `temporal-splatfacto`

`temporal-splatfacto` is a new nerfstudio method that reconstructs a **moving
scene observed by fixed cameras** as one Gaussian-Splatting cloud *per video
frame*. It is a nerfstudio port of the 4DGS thesis script `4dgs/train_video.py`,
re-expressed as a first-class `ns-train` method so it gets the live viser viewer,
nerfstudio checkpoints, and `ns-export` for free.

## Why this works on the `static*` cameras (and not `dynamic1`)

The room dataset has four **static** cameras (`output/static1` … `output/static4`).
Each is a *fixed* viewpoint (every pose in its `transforms.json` is identical)
that records one image per time step (`images/0001.png` … `images/0150.png`).
The mannequin/scene moves; the cameras do not. So:

- **Across cameras** (static1..4) we get the 4-view parallax needed to fit a 3D
  cloud — exactly what a single fixed view (`dynamic1`) can never provide (see
  the note in section 0 about `dynamic1` having no parallax).
- **Across images** (`0001`→`0150`) we get the temporal sequence: image index
  *t* of every camera = frame *t* of the reconstruction.

`temporal-splatfacto` uses **only** the `static*` cameras (the
`--pipeline.datamanager.dataparser.camera-prefix` is `static`), deliberately
ignoring `dynamic1`.

## Input data format = the `output/` format

The method reads the **same data layout that lives in `output/`** — one
nerfstudio dataset per camera, each a folder with `transforms.json` (explicit
`fl_x, fl_y, cx, cy, w, h`) and an `images/` directory. No conversion step is
needed; point `--data` at the parent `output/` directory and the temporal
dataparser discovers every `static*` sub-folder, aligns their frames by sorted
filename, and flattens them into one time-indexed dataset.

```
output/                      <-- pass this as --data
├── static1/
│   ├── transforms.json      # constant pose + intrinsics
│   └── images/
│       ├── 0000.png         # OPTIONAL empty-room background plate (see below)
│       └── {0001..0150}.png # the moving subject, one per time step
├── static2/ ...
├── static3/ ...
├── static4/ ...
└── dynamic1/                # ignored (prefix != "static")
```

The subject must arrive **segmented onto a black background** (the visual hull, the
foreground masking, and the black training background all assume it). There are two ways
to get there, selected automatically by the data layout:
- **Pre-segmented images** — `images/` already holds the subject on black (the legacy
  `output/` layout, frames `0001..0150` only). Nothing extra happens.
- **Background subtraction** — `images/` holds *un-segmented* room photos plus a frame
  `0000.png`, and the method segments the subject itself, with no segmentation model. See
  the next section.

## Object segmentation by background subtraction (no segmentation model)

The whole temporal pipeline assumes the moving subject is **segmented onto a black
background**. Rather than depend on a learned segmentation model, the dataparser can recover
the silhouette itself from a **background plate** — a single photo of the *empty room*,
without the subject, taken from each fixed camera before the subject enters.

### Idea

A static camera that films a moving subject sees the *same* background in every frame; only
the subject changes. So if we have one frame of the empty room (the **plate**), the subject
in any later frame is exactly **where that frame differs from the plate**. For every pixel we
compute the colour-difference magnitude against the plate; where it is large the scene
*changed*, so the subject is there and the pixel is kept; where it is ~unchanged the pixel is
**set to black** (the renderer's default background). No network, no training, no labels —
just a per-pixel difference and a threshold. This is classic background subtraction, and it
is exact for a static rig.

### Trigger (frame `0000`)

Segmentation runs **only when every camera provides a frame numbered `0`** (e.g.
`images/0000.png`). That frame is taken to be the empty-room plate. When no `0000.png` is
present the images are assumed to be already segmented and this step is skipped, so the legacy
`output/` layout (frames `0001..0150`) is unaffected. If only *some* cameras have a `0000.png`
the run aborts with a clear error (give every camera a plate, or none).

The plate is **consumed** by segmentation and is *not* a reconstruction frame: with a plate
plus `0001..0150` the reconstruction still has 150 frames (`0001` becomes reconstruction frame
0), and `max_num_iterations` is sized accordingly.

### What it does, per camera

For each later frame `I_t` and the plate `I_0` (same camera):

1. **Change magnitude** `d = mean_RGB(|I_t − I_0|)` per pixel (on a 0–1 scale).
2. **Threshold** `mask = d > bg_subtract_threshold` (default `0.1`) — the changed (object)
   pixels.
3. **Morphological opening** (`bg_subtract_open`, default radius 1 px) removes isolated
   speckle — antialiasing along edges, faint shadow / indirect-light flicker on the walls and
   floor.
4. **Dilation** (`bg_subtract_dilate`, default 2 px) grows the mask slightly so soft object
   edges are kept, not clipped.
5. **Black out** everything outside the mask: `I_t ← I_t · mask`.

The segmented frames are cached under `images_bgsub/` inside each camera folder (re-used across
runs unless the parameters or the frame set change), and the dataparser then points training at
those instead of the originals — so the visual hull, the brightness foreground masks, and the
per-frame hull snapping all operate on the segmented images with no further changes. (You may
want to add `images_bgsub/` to your `.gitignore`.)

### A note on shadows / indirect light

Background subtraction attributes *any* change to the subject, including the soft shadow it
casts and the indirect light it bounces onto static surfaces. Two things keep this benign here:
the morphological opening removes the thin, low-contrast shadow fringe, and — more importantly
— the downstream **visual-hull intersection across all 4 cameras** keeps only voxels that are
foreground in *every* view, so a shadow that leaks into one camera's segmentation is carved
away unless all four cameras agree on it (which a cast shadow on the floor does not). Raise
`bg_subtract_threshold` if low-contrast shadows still leak through.

### Disable / tune

```bash
# never segment, even if a 0000.png plate is present (treat 0000 as a normal frame):
--pipeline.datamanager.dataparser.bg-subtract False
# stricter (kills more shadow leak, may eat dim object parts):
--pipeline.datamanager.dataparser.bg-subtract-threshold 0.15
```

### Shadow-aware hysteresis segmentation (`bg-subtract-mode hysteresis`)

The single global threshold has a failure mode no value of it can fix: **camouflage**.
Where the subject's colours locally match the background (the mannequin's legs over the
rug in the early frames), the plate-difference is *weak but nonzero* — lowering the
threshold enough to admit it also admits the cast shadow and bounce light, which live in
the same magnitude band. The `hysteresis` mode separates the two kinds of weak evidence
by the two properties magnitude discards:

1. **Photometric signature (shadow veto).** A cast shadow *scales illumination down*:
   luminance drops, chromaticity (colour direction, `I/lum(I)`) stays put. A weak pixel
   is kept only if its chromaticity moved by more than `bg-subtract-chroma-threshold`
   vs the plate, or it got *brighter* (which a shadow cannot do).
2. **Connectivity (hysteresis).** Pixels with `d > bg-subtract-threshold` (the *high*
   threshold in this mode) are kept unconditionally; surviving weak pixels
   (`d > bg-subtract-low-threshold`) join only if geodesically **connected** to strong
   evidence (reconstruction-by-dilation to a fixpoint, run on the GPU). Camouflaged
   body parts are attached to the confidently detected torso and get in; free-floating
   bounce-light islands don't.

Measured against the ground-truth silhouettes over all 150 frames x 4 cameras
(`experiments/seg_full_eval.py`): **IoU 0.962 / precision 0.986 / recall 0.975 /
segmented-image PSNR 33.8 dB**, vs 0.863 / 0.935 / 0.918 / 30.6 dB for the tuned
single-threshold production setting — and the *worst* frame improves from IoU 0.77 to
0.91 (the camouflage window is essentially healed; see
`experiments/outputs/bgsub/seg_variants.json` for the variant sweep + ablations:
hysteresis *without* the shadow veto floods through the contact shadow, precision 0.37;
the veto *without* connectivity reaches IoU 0.942 — both parts needed). The result is
also insensitive to the high threshold (identical IoU for tau_hi in 0.08–0.12), which
removes the pipeline's most delicate tuning knob.

```bash
--pipeline.datamanager.dataparser.bg-subtract-mode hysteresis \
--pipeline.datamanager.dataparser.bg-subtract-threshold 0.1 \
--pipeline.datamanager.dataparser.bg-subtract-dir images_bgsub_hyst
```

(Use a dedicated `bg-subtract-dir` per mode so caches don't clobber each other.)

## The algorithm and the optimizations implemented

Training is a single, monotonically increasing step sequence partitioned into
one **initial** block (frame 0) followed by one **tracking** block per
subsequent frame:

```
| <----- initial_iterations -----> | <- track -> | <- track -> | ...
|              frame 0              |   frame 1   |   frame 2   | ...
```

0. **Visual-hull initialization (before frame 0).** Because the object is segmented
   onto a black background (either pre-segmented, or by the
   [background subtraction](#object-segmentation-by-background-subtraction-no-segmentation-model)
   above), the gaussians are seeded from the object's **visual
   hull** rather than a random cloud. Each frame-0 camera image is thresholded by
   brightness into a foreground mask; a voxel grid over the scene is then *carved*
   by keeping only voxels that project into the foreground of **every** camera —
   the intersection of the four silhouette cones, i.e. a bounded volume that hugs
   the object seen from all sides. The surviving voxels seed the gaussians, with
   colours sampled from the images. So from step 0 the gaussians sit only inside
   the thing being reconstructed (no stray background gaussians to cull later).
   This is done by `TemporalDataParser` and handed to the model through
   nerfstudio's normal seed-points path. Disable with
   `--pipeline.datamanager.dataparser.init-visual-hull False`.

1. **Frame 0 — dense reconstruction.** Normal splatfacto with densification for
   `initial_iterations` steps, starting from the hull seed. This both refines the
   geometry *and fixes the gaussian count* for the rest of the video
   (densification/culling is hard-stopped at `initial_iterations` via the gsplat
   strategy's `refine_stop_iter`). The training **background is black**
   (`background_color = "black"`) to match the segmented inputs, so empty regions
   cost nothing in the loss. The loss is also **foreground-masked with silhouette
   supervision** (see [Preventing background drift](#preventing-background-drift-foreground-masking--silhouette-supervision)),
   which keeps gaussians on the object instead of letting them fit the black
   background and drift over the video.

2. **Frames 1..T — tracking.** Each subsequent frame is refined for
   `tracking_iterations` steps with the gaussian count frozen. Three
   optimizations (all from `train_video.py`) speed up and stabilize tracking:

   - **Linear velocity prediction (mask-gated).** At every frame transition the
     gaussian positions are extrapolated `x_t += (x_{t-1} - x_{t-2})`, so a smoothly
     moving scene starts each frame already close to the answer. Disable with
     `--pipeline.model.use-velocity-prediction False`. Drifted gaussians are then
     **snapped back onto the object** each frame to prevent *runaway velocity* and
     reclaim *left-behind* gaussians — see
     [Runaway velocity & left-behind gaussians](#runaway-velocity--left-behind-gaussians-per-frame-hull-snapping)
     below.
   - **Constant count & colour (the core invariants).** After frame 0 the gaussian
     **count is fixed** (densification/culling hard-stopped) and the per-gaussian
     **colour is frozen** (`features_dc`, `features_rest`; `requires_grad=False`).
     With appearance and population fixed, the only way the optimizer can reduce the
     loss is to **move/reshape** the gaussians — so tracking aligns them to the
     object's motion instead of re-fitting appearance. (Verified: across frames the
     count stays at e.g. 1620 and the colour tensors are bit-identical, while the
     means keep moving.) Disable colour freezing with
     `--pipeline.model.freeze-colors-when-tracking False`. See
     [Preventing background drift](#preventing-background-drift-foreground-masking--silhouette-supervision)
     for why opacity is deliberately **not** frozen.
   - **Per-phase means learning rate.** Frame 0 decays the means LR
     (`means_lr_init` → `means_lr_final`); tracking holds it constant at
     `tracking_means_lr`. (A single global decay — fine for one static scene —
     would shrink the LR to nothing by the late frames and freeze the motion.)

   Resolution is kept full (`num_downscales = 0`) so every frame trains at the
   same resolution.

## Run it — `ns-train temporal-splatfacto`

```bash
CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/ns-train temporal-splatfacto \
  --data ../output \
  --output-dir splats/temporal \
  --vis viewer \
  --viewer.quit-on-train-completion False \
  --viewer.websocket-port 7007 \
  --experiment-name temporal \
  --pipeline.model.initial-iterations 3000 \
  --pipeline.model.tracking-iterations 300
```

- **`--data ../output`** — parent folder containing the `static*` datasets.
- **`max_num_iterations` is auto-computed.** You do *not* set it; the temporal
  trainer counts the frames and sets it to
  `initial_iterations + (num_frames - 1) * tracking_iterations` so the run ends
  exactly when the last frame is done. With 150 frames and the defaults above
  that is `3000 + 149*300 = 47700` steps.
- **No COLMAP / point cloud required** — random camera-bbox init is used.

**Open the viewer at** http://localhost:7007 (forward the port over SSH if
remote). The reconstruction updates live, so you watch the cloud snap to each
frame as training advances.

### Which frame is currently being aligned

The viser viewer shows a read-only **"Temporal frame"** field (under *Custom
Elements*) that updates every step, e.g.:

```
frame 42/150 · tracking (step 15300)
```

The same information is logged to the console at each frame transition (velocity
magnitude) and each frame save.

## Checkpoints — per-frame `.ply` + nerfstudio `.ckpt`

Two complementary artifacts are produced:

1. **Per-frame `.ply` (primary temporal output).** The converged gaussians of
   **every frame** are written to:

   ```
   splats/temporal/temporal/temporal-splatfacto/<timestamp>/temporal_frames/
   ├── 00000.ply
   ├── 00001.ply
   └── ... (one per frame, 00149.ply)
   ```

   These use the exact same property layout as `ns-export gaussian-splat`
   (`sh_coeffs` mode), so each loads in any nerfstudio/Inria 3DGS viewer. This is
   the natural "video" output — play the plys in sequence. Disable with
   `--pipeline.model.save-ply-per-frame False`.

2. **Nerfstudio `.ckpt` (resumable state).** Standard nerfstudio checkpoints are
   still written every `steps_per_save` steps to `nerfstudio_models/`, capturing
   the full pipeline (the gaussians of whichever frame was current at save time)
   plus optimizer state.

### View / export afterwards

```bash
# Re-open the viewer for the finished run (shows the last frame's cloud):
.venv/bin/ns-viewer --load-config splats/temporal/temporal/temporal-splatfacto/<timestamp>/config.yml

# Export the current (last-frame) cloud via the standard exporter:
.venv/bin/ns-export gaussian-splat \
  --load-config splats/temporal/temporal/temporal-splatfacto/<timestamp>/config.yml \
  --output-dir splats/temporal/export
```

(For per-frame results, just use the `temporal_frames/*.ply` written during
training — no extra export step needed.)

## Play it back — the temporal viewer (`experiments/temporal_viewer.py`)

`ns-viewer` only ever shows **one** static cloud (the last frame). To actually
*watch the animation* — and to **composite the static background with the moving
foreground** — use the dedicated playback viewer. It loads the per-frame
`temporal_frames/*.ply` (the dynamic foreground) and, optionally, a static
background `.ply` (e.g. a `splatfacto` export), renders them together in a
**viser** web viewer, and gives you transport controls.

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python experiments/temporal_viewer.py \
  splats/temporal/temporal/temporal-splatfacto/<timestamp>/temporal_frames \
  --fps 30 \
  --port 7007
```

You can also pass the **run directory** instead of `temporal_frames/` — it finds
the `temporal_frames/` subfolder automatically.

**Composite a static background** (so the moving character plays *over* a fixed
room reconstructed with plain `splatfacto`):

```bash
.venv/bin/python experiments/temporal_viewer.py \
  splats/temporal/temporal/temporal-splatfacto/<timestamp>/temporal_frames \
  --background splats/dynamic/export/splat.ply \
  --fps 30
```

**Open the viewer at** http://localhost:7007 (forward the port over SSH if
remote). The right-hand panel has, under *Temporal playback*:

- **`▶ Play` / `⏸ Pause`** — start/stop automatic playback at the chosen FPS.
- **`Loop`** — when on, wraps from the last frame back to the first; when off,
  playback stops on the last frame.
- **`FPS`** — playback framerate (1–60), adjustable live.
- **`Frame`** — drag to scrub to any frame manually (this also pauses playback);
  the **`◀ Prev` / `Next ▶`** buttons step one frame at a time when paused.
- **`Showing`** — read-only `frame i / N` indicator.

### How it loads frames (all-in-memory, instant switching)

Every frame is read, activated (opacity `sigmoid`, scale `exp`, quaternion
normalised, colour `0.5 + SH_C0·f_dc`), and uploaded to the browser **once at
startup** as its own Gaussian-splat scene node — all hidden except frame 0. The
static background, if given, is added once and stays visible behind every frame.
Stepping or playing then only flips each node's `visible` flag, so there is **no
per-frame disk read, decode, or re-upload during playback** — it stays smooth
even at high FPS. The 4DGS output holds the gaussian count constant (~a few
thousand) across frames, so keeping all 150 frames resident is cheap (tens of MB
of host RAM, a few hundred MB in the browser).

> For sequences far larger than this room dataset (thousands of frames), a
> windowed loader — keep N frames resident and stream more during playback —
> would be the way to bound memory; it is not implemented because full preload is
> strictly faster and well within budget for the current data.

| Flag | Default | Description |
|---|---|---|
| `frames_dir` (positional) | (required) | `temporal_frames/` dir, or a run dir that contains it |
| `--background` | `None` | Optional static background `.ply` composited behind the animation |
| `--fps` | `30` | Initial playback framerate (also adjustable in the GUI) |
| `--port` | `7007` | viser websocket port |
| `--max-frames` | `-1` | Cap the number of frames loaded (`-1` = all) |

The viewer reads the **same per-frame ply layout** the trainer writes (and that
`temporal_compress.py` round-trips), so a `frames_restored/` directory from
`temporal_compress.py decompress` plays back identically.

## All temporal options

| Flag | Default | Description |
|---|---|---|
| `--data` | (required) | Parent dir containing the `static*` camera datasets |
| `--pipeline.model.initial-iterations` | `3000` | Frame-0 reconstruction steps (with densification) |
| `--pipeline.model.tracking-iterations` | `300` | Steps per subsequent frame (no densification) |
| `--pipeline.model.use-velocity-prediction` | `True` | Linear `x_t += x_{t-1} - x_{t-2}` extrapolation between frames |
| `--pipeline.model.snap-strays-to-hull` | `True` | Each frame, snap gaussians outside the mask-intersection back onto the hull (reclaims strays, fixes runaway) |
| `--pipeline.model.snap-min-view-fraction` | `1.0` | Snap a gaussian if inside fewer than this fraction of masks (`1.0` = not inside all) |
| `--pipeline.model.snap-distance-margin` | `0.0` | Distance-field clamping: snap a flagged gaussian only if it is farther than this many hull-voxel sizes from the hull (0 = snap all flagged) |
| `--pipeline.model.temporal-smoothness-lambda` | `0.0` | L2 anchor pulling opacity/scale/quat towards their previous-frame converged values during tracking (0 = off) |
| `--pipeline.model.hull-snap-resolution` | `96` | Voxel grid resolution for the per-frame hull used as snap targets |
| `--pipeline.model.velocity-mask-gating` | `True` | Fallback when snapping is off: zero stray velocity (stops coasting but leaves behind) |
| `--pipeline.model.velocity-gate-min-view-fraction` | `0.5` | (Gating only) fraction of cameras a gaussian must project onto to keep its velocity |
| `--pipeline.model.freeze-colors-when-tracking` | `True` | Freeze SH colours after frame 0 (constant colour → motion-only tracking) |
| `--pipeline.model.freeze-opacity-when-tracking` | `False` | Also freeze opacity (opt-in; tested to worsen background fitting — see below) |
| `--pipeline.model.tracking-means-lr` | `1.6e-4` | Constant means LR during tracking |
| `--pipeline.model.means-lr-init` / `--...-final` | `1.6e-4` / `1.6e-6` | Means LR decay range during frame 0 |
| `--pipeline.model.save-ply-per-frame` | `True` | Write `temporal_frames/<frame>.ply` for each frame |
| `--pipeline.model.background-color` | `black` | Training/render background (`black` matches the segmented inputs) |
| `--pipeline.model.use-foreground-mask` | `True` | Restrict the photometric loss to the object silhouette |
| `--pipeline.model.alpha-mask-loss-lambda` | `1.0` | Weight of the α→0-outside silhouette penalty (the anti-drift term) |
| `--pipeline.model.alpha-inside-loss-lambda` | `0.1` | Weight of the gentle α→1-inside (keep object solid) term |
| `--pipeline.model.mask-threshold` | `0.05` | GT brightness above which a pixel is object foreground |
| `--pipeline.model.mask-dilate` | `3` | Silhouette dilation (px); defines the don't-care band at the edge |
| `--pipeline.datamanager.dataparser.bg-subtract` | `True` | Segment the subject by background subtraction when a `0000.png` plate is present (no model) |
| `--pipeline.datamanager.dataparser.bg-subtract-mode` | `threshold` | `threshold` = single global threshold; `hysteresis` = shadow-aware two-band segmentation (see above) |
| `--pipeline.datamanager.dataparser.bg-subtract-low-threshold` | `0.01` | (hysteresis) weak-evidence floor |
| `--pipeline.datamanager.dataparser.bg-subtract-chroma-threshold` | `0.04` | (hysteresis) min chromaticity change for a darkened weak pixel to count as object, not shadow |
| `--pipeline.datamanager.dataparser.bg-subtract-threshold` | `0.06` | Per-pixel change magnitude (mean abs RGB diff vs the plate) above which a pixel is object (the *high* threshold in hysteresis mode, where `0.1` is used) |
| `--pipeline.datamanager.dataparser.bg-subtract-open` | `1` | Morphological-opening radius (px) to remove speckle/shadow fringe (0 = off) |
| `--pipeline.datamanager.dataparser.bg-subtract-dilate` | `2` | Dilate the final object mask by N px so soft edges are kept |
| `--pipeline.datamanager.dataparser.init-visual-hull` | `True` | Seed gaussians from the carved visual hull instead of randomly |
| `--pipeline.datamanager.dataparser.hull-brightness-threshold` | `0.05` | Brightness (0-1) above which a pixel is object foreground |
| `--pipeline.datamanager.dataparser.hull-grid-resolution` | `192` | Voxels per axis of the carving grid (higher = denser hull) |
| `--pipeline.datamanager.dataparser.hull-dilate` | `2` | Dilate each mask by N px before carving (forgiving at edges) |
| `--pipeline.datamanager.dataparser.hull-min-view-fraction` | `1.0` | Fraction of cameras a voxel must be foreground in (`1.0` = strict intersection) |
| `--pipeline.datamanager.dataparser.hull-cube-scale` | `1.0` | Carving-cube side as a multiple of the camera bbox |
| `--pipeline.datamanager.dataparser.hull-max-points` | `300000` | Subsample the hull down to at most this many seeds |
| `--pipeline.datamanager.dataparser.camera-prefix` | `static` | Only folders with this prefix are used as cameras |
| `--pipeline.datamanager.dataparser.max-frames` | `-1` | Cap the number of temporal frames (`-1` = all) |
| `--pipeline.datamanager.camera-res-scale-factor` | `1.0` | Resize images **and** intrinsics on load (e.g. `0.5` = half resolution) |
| `--pipeline.datamanager.cache-images` | `cpu` | `cpu` or `gpu` (the schedule needs deterministic indexing; `disk` is not supported) |

Note: with 4 cameras × N frames the datamanager caches `4·N` images. At full
resolution that is large, so `cache_images='cpu'` with `cache_images_type='uint8'`
are the defaults (≈ 3.7 GB CPU RAM for 150 frames of 1080p). Use
`--pipeline.datamanager.camera-res-scale-factor 0.5` to halve the resolution
(and quarter the memory) of both the cached images and the cameras.

**Tuning the visual hull.** On startup the carve logs how many seed gaussians it
produced, e.g. `visual hull: 1620 seed gaussians carved from 4 cameras`. If that
number is **too low** (or it warns the hull is empty), the masks are probably too
small — lower `hull-brightness-threshold`, raise `hull-dilate`, or relax
`hull-min-view-fraction` (e.g. `0.75` so a voxel need only be foreground in 3 of 4
cameras). For a **finer** hull raise `hull-grid-resolution` (cost grows ~cubically).
Frame-0 densification then adds detail on top of the seed, so a moderately sparse
hull is fine.

## Preventing background drift: foreground masking & silhouette supervision

### Symptom

Over the video the gaussians **drift off the object and start fitting the black
background** — the object gets messier and faint splats accumulate in empty space.

### Root cause (and how it relates to the original `train_video.py`)

The original `4dgs/train_video.py` masks the loss by the image's **alpha channel**
(`image = image * cam.alpha_mask`, where `alpha_mask` is the 4th channel of an RGBA
image — `scene/cameras.py:46`). Its segmentation lived in alpha, so the background
never contributed to the loss.

Our renders in `output/static*/images/` are RGBA too, **but their alpha is fully
opaque (all 1.0)** — the segmentation is baked into the RGB as a black background, not
into alpha (verified: `alpha.min()==alpha.max()==1.0`). So a 1-to-1 port of the
original loss masks by an all-ones mask, i.e. **masks nothing**, and the full image
(97% black background) drives the loss. With a black background:

- A bright, object-coloured gaussian that wanders into the background renders bright on
  black → large error → the optimizer drives its **opacity to ~0** rather than moving it
  back. It becomes an invisible "dead" gaussian that keeps getting velocity-pushed each
  frame and accumulates in the background; the object loses it.
- Matching black-on-black costs ~0, so there is **no force pinning gaussians to the
  object silhouette**. Drift compounds frame over frame.

(The training background being **black** is confirmed used for the standard command —
`background_color` defaults to `black` and the command does not override it. That alone
is necessary but not sufficient; the missing piece is the masking the original had.)

### The fix

Re-introduce the original's masking — but derive the mask from **brightness** (since our
alpha is useless) — and add explicit **silhouette (alpha) supervision**, which the
original did not need because its mask did the job implicitly:

1. **Foreground-masked photometric loss.** The L1 + SSIM terms are computed on the
   object region only (rendered & GT multiplied by the object mask), exactly like the
   original's `image * alpha_mask`. The mask is `GT brightness > mask_threshold`.
2. **Silhouette / alpha supervision (the actual anti-drift term).** The rendered
   *accumulation* (opacity) is pushed towards **0 clearly outside** the object and gently
   towards **1 in the object core**. Now a gaussian that wanders into the background
   renders nonzero opacity there and is **directly penalized** → pushed back onto the
   object or faded out. A "don't-care" band (between the raw silhouette and its dilation)
   around the edge is neither pushed to 0 nor 1, so gaussian tails at the true boundary
   are not fought.

Implemented in `TemporalSplatfactoModel.get_loss_dict` /`.foreground_masks`
(`models/temporal_splatfacto.py`); on by default. Applies to frame 0 and every tracking
frame.

### Measured effect

A 16-frame tracking run, measuring per frame the mean rendered opacity that leaks into
the clearly-background region (rising = drift), masking **off vs on**:

| | mean background leak | trend over 16 frames | object core coverage |
|---|---|---|---|
| **off** (full-image loss) | 0.0059 | grows ~100× (0.0001 → 0.0127) | degrades to ~0.82 |
| **on** (masked + α supervision) | **0.0020** | stays flat & low (≤ 0.0033) | holds ~0.86–0.95 |

So background fitting is suppressed ~3× on average (and its frame-over-frame *growth* —
the actual drift — is essentially eliminated), while the object stays more solid.

### Why not just freeze opacity too? (tested — it backfires)

A natural idea, given the constant-count/constant-colour philosophy, is to also **freeze
opacity** so gaussians can *only move* and cannot "hide" by fading into the background.
We tested it (`--pipeline.model.freeze-opacity-when-tracking True`) against the default
silhouette approach over 16 frames:

| strategy | mean background leak | mean object coverage | leak trend |
|---|---|---|---|
| baseline (full-image loss, opacity trainable) | 0.0056 | 0.885 | grows (drift) |
| **silhouette loss, opacity trainable** *(default)* | **0.0020** | **0.933** | flat & low |
| freeze opacity **and** colour | 0.0123 | 0.888 | grows worst (→0.044) |

Freezing opacity made background fitting **~6× worse**. The reason is the interaction with
the masked loss: the photometric loss is masked to the object, so a gaussian that gets
pushed into the background has **no photometric pull-back** there — and freezing opacity
removes its only other escape (fading out), so it stays **bright and opaque in the
background**. Keeping opacity *trainable* is precisely what lets the silhouette (alpha)
loss fade such gaussians away. So colour and count are held constant, but **opacity is
deliberately left free** (this also matches the original `train_video.py`, which trains
opacity during tracking). The opt-in flag exists for experimentation but is not
recommended on this data.

### Tuning

- Still seeing faint background splats → raise `--pipeline.model.alpha-mask-loss-lambda`
  (e.g. `2.0`).
- Object looks eaten at the edges → raise `--pipeline.model.mask-dilate` or lower
  `--pipeline.model.mask-threshold` (grow the mask).
- Object too transparent → raise `--pipeline.model.alpha-inside-loss-lambda`.
- Disable entirely with `--pipeline.model.use-foreground-mask False
  --pipeline.model.alpha-mask-loss-lambda 0`.

## Runaway velocity & left-behind gaussians: per-frame hull snapping

### Symptom

A gaussian that moves between frame 0 and frame 1 (e.g. upward) keeps moving in that
direction at constant speed for the rest of the video, drifts off the object, its opacity
drops, and it never comes back — it no longer corresponds to the character. Over many frames
the object loses gaussians and degrades.

### Root cause

Velocity prediction sets `x_t += (x_{t-1} - x_{t-2})` each frame, where `x_{t-1}`, `x_{t-2}`
are the gaussian's *converged* positions in the two previous frames. This is self-correcting
**only while the gaussian still receives a gradient**: normally the per-frame optimization
pulls it onto the object, so its converged position reflects the true motion and the
estimated velocity stays accurate. But once a gaussian drifts off the object it **fades**
(the photometric loss is masked to the object, so off-object pixels give no gradient) and its
position stops being corrected. Its frame-to-frame displacement freezes at a constant value,
so the extrapolated velocity freezes too, and the gaussian **coasts away in a straight line
forever** — runaway velocity — and is lost from the reconstruction.

### The fix — snap strays back onto the visual hull every frame

At the start of every frame, **after** applying velocity, every gaussian center is projected
into that frame's cameras. Any gaussian that lands **outside the mask-intersection (the visual
hull)** is snapped onto its **nearest hull-surface voxel**, so the frame begins with all
gaussians inside the object as seen from all cameras — they *must* align to it. This
**reclaims** drifted gaussians instead of abandoning them (and also corrects velocity
under/overshoot, since a cloud that lags the moving object gets snapped forward onto it). The
velocity history of snapped gaussians is reset so the teleport is not counted as motion.

The hull is carved each frame on a voxel grid (`hull_snap_resolution`) spanning the current
cloud, keeping voxels inside every camera mask; strays are matched to it by nearest neighbour.
"Stray" defaults to *not inside every mask* (`snap_min_view_fraction = 1.0`, the strict
intersection you'd expect from "inside all camera object maps"). Implemented in
`TemporalSplatfactoModel._snap_strays_to_hull` / `step_cb` (`models/temporal_splatfacto.py`);
on by default. Each frame logs e.g. `snapped 41/1620 strays onto the visual hull`.

A milder alternative, **velocity gating** (`velocity_mask_gating`), merely *zeroes* a stray's
velocity so it stops coasting but stays where it is (left behind, fading). Snapping supersedes
it and is used by default; gating is the fallback when snapping is disabled.

### Measured effect

16-frame tracking run — gating (freeze & leave behind) vs snapping (reclaim):

| | mean object coverage | mean object PSNR | final cloud max radius |
|---|---|---|---|
| velocity gating | 0.893 (degrades 1.00 → 0.78) | 13.24 dB (→ 10.6) | 0.24 |
| **hull snapping** *(default)* | **0.971** (holds 1.00 → 0.94) | **14.27 dB** (→ 12.1) | **0.19** |

Snapping keeps the object solid and higher-quality, and the advantage **widens over frames**
(snapping is ~+1.5 dB and +0.16 coverage by frame 15) — i.e. it directly counteracts the
left-behind degradation. The cloud also stays tight (no runaway).

### Distance-field clamping (`snap-distance-margin`)

The strict projective stray test is **binary in exactly the way the masks are not
reliable**: a gaussian one pixel outside a noisy mask boundary (or in a residual
silhouette hole) fails it just like one that coasted ten subject radii away — and on
imperfect masks it flags a large fraction of the cloud every frame (~half on the
flagship run), teleporting locally-converged geometry that the next tracking block must
rebuild. With `--pipeline.model.snap-distance-margin D` (in hull-voxel units, e.g.
`2.0`), the projective test only *nominates* candidates; a candidate is snapped only if
its 3D distance to the nearest hull voxel exceeds `D` voxel sizes. Geometrically the
hull gets a margin shell: borderline gaussians inside the shell are left to the
photometric optimizer (which, unlike the binary test, has gradient information), true
runaways beyond it are still reclaimed. Costs nothing extra — the nearest-voxel distance
is already computed to find the snap target. Per-frame `flagged` vs `snapped` counts are
written to `temporal_frames/temporal_stats.json` for any run.

### Temporal attribute smoothness (`temporal-smoothness-lambda`)

Between the frozen (colour, count) and the moving (positions) parameters sit opacity,
scale and rotation: trainable by necessity, but for a rigid subject their *true*
per-frame evolution is small. A generous tracking budget lets SGD wander them around
their optimum, so their converged per-frame values **jitter** — invisible per frame,
but it flickers in playback and fills the attribute time series with high-frequency
energy that the DCT compressor cannot truncate (its opacity/scale/rot channels saturate
the coefficient cap). `--pipeline.model.temporal-smoothness-lambda L` adds an L2 anchor
pulling opacity logits, log-scales and quats towards their **previous frame's converged
values** during tracking (positions exempt — their change is the genuine motion). This
is the compressor's smoothness prior moved into the optimizer, where it damps the noise
at the source. `L=0.3` was selected by a 40-frame pilot sweep over {0.3, 1.0, 3.0}:
jitter suppression saturates at the smallest weight while the structural cost keeps
growing with `L` (object SSIM 0.827 -> 0.814/0.799/0.789 for L=0.3/1/3). On the full
150-frame run at L=0.3: opacity jitter 0.224 -> 0.0012 logits/frame (~180x), scale ~78x,
rotation ~5x, position (deliberately exempt) unchanged; compression at q=0.999 improves
27.8x@26.2dB -> 43.5x@27.0dB (opacity/scale collapse from the 64-coeff cap to K<=2;
quats still cap). Solo cost is SSIM 0.833 vs 0.854 — absorbed entirely in the combined
`imp_full` config (0.858). Object PSNR unaffected in all arms.

### Tuning

- Snap only clearly-drifted gaussians (gentler, leaves dark in-object regions alone) → lower
  `--pipeline.model.snap-min-view-fraction` toward `0.5` (snap only those outside a majority of
  masks).
- Finer / coarser snap targets → raise / lower `--pipeline.model.hull-snap-resolution`.
- Disable snapping with `--pipeline.model.snap-strays-to-hull False` (falls back to velocity
  gating; disable that too with `--pipeline.model.velocity-mask-gating False`).

## Early stopping of per-frame tracking

A tracking frame can be stopped as soon as its reconstruction has converged, instead
of always spending the full `tracking-iterations`. Convergence is decided by an
**object-region PSNR** criterion (PSNR measured only on foreground pixels — the black
background is most of the frame and would otherwise swamp the metric): a frame is done
once its PSNR stops improving by more than `early-stop-min-delta` dB for
`early-stop-patience` consecutive steps (a plateau), or reaches `early-stop-target-psnr`
if set. Once converged, the frame's gaussians are frozen (geometry LRs → 0) for the rest
of its budget. Enable with:

```bash
--pipeline.model.early-stop-enabled True \
--pipeline.model.early-stop-patience 20 \
--pipeline.model.early-stop-min-delta 0.1
# or stop at an absolute quality:
--pipeline.model.early-stop-target-psnr 25.0
```

| Flag | Default | Description |
|---|---|---|
| `--pipeline.model.early-stop-enabled` | `False` | Turn on per-frame early stopping during tracking |
| `--pipeline.model.early-stop-patience` | `30` | Steps without a PSNR improvement before declaring convergence |
| `--pipeline.model.early-stop-min-delta` | `0.03` | Min object-PSNR (dB) increase that counts as improvement |
| `--pipeline.model.early-stop-target-psnr` | `None` | If set, stop at this absolute object PSNR instead of a plateau |
| `--pipeline.model.early-stop-fg-threshold` | `0.05` | GT brightness above which a pixel is object foreground |

### Experiment: does velocity prediction reduce steps-to-convergence?

`experiments/velocity_early_stopping.py` uses the early-stopping metric to quantify the
benefit of velocity prediction. It fits frame 0 once, then tracks every frame to
convergence **with** and **without** velocity prediction from the identical starting
point, recording the steps each frame needs, and plots the per-frame distribution as
overlaid histograms with the means marked.

```bash
CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_VISIBLE_DEVICES=4 \
  .venv/bin/python experiments/velocity_early_stopping.py \
    --data ../output/dataset --num-frames 40 --target-psnr 20.0 --camera-res-scale-factor 0.25 \
    --initial-iterations 3000 --snap-distance-margin 2.0 \
    --bg-subtract-mode hysteresis --bg-subtract-threshold 0.1 --bg-subtract-dir images_bgsub_hyst \
    --output-dir experiments/outputs/velocity_target
```

Outputs `velocity_convergence_histogram.png` and a summary JSON into `--output-dir`
(`experiments/outputs/velocity_target` and `velocity_plateau` are what `make_figures.py`
reads). The script accepts the improved-pipeline knobs (`--bg-subtract-mode`,
`--snap-distance-margin`, `--temporal-smoothness-lambda`). Use `--target-psnr X` for a
same-quality comparison (steps to reach X dB — the fairest reading of "same metric
value"), or omit it for the plateau criterion. **Result on the room data** (2026-07-06
rerun under the improved pipeline: hysteresis masks, margin 2.0, N0=3000; 40 frames,
target = 20 dB object PSNR): velocity prediction reached the target in a mean of
**23 steps/frame** vs **145 steps/frame** without it — about **84% fewer steps** — and
every frame reached the target under both conditions (max 59 vs 282 steps, none hit the
400-step cap). With the plateau criterion instead, velocity also converges to
substantially higher quality: **29.3 dB vs 23.4 dB**, because without a warm start the
per-frame fit drifts and plateaus at a worse solution.

## Temporal compression of the per-frame `.ply` files

`experiments/temporal_compress.py` compresses a directory of temporal frame plys (the
`temporal_frames/` output) into a much smaller directory, and decompresses it back into the
identically-numbered frame plys.

### Idea

Every frame shares the **same gaussians in the same order** (count and colour are held
constant; only motion/shape/opacity change). So each per-gaussian attribute — `x`, `y`, `z`,
`opacity`, `scale_*`, `rot_*` — is a **time series** of length F (number of frames), and most of
these series are very smooth or constant. We compress each one in a frequency basis:

- **Identically-zero** attributes (`nx, ny, nz` normals) → stored not at all.
- **Constant-in-time** attributes (frozen colours `f_dc_*`) → stored once per gaussian.
- **Time-varying** attributes → **DCT along time, keep the lowest `K` coefficients** per
  attribute (reconstruct by inverse DCT with the dropped coefficients zeroed — a smooth
  low-pass fit to the attribute's evolution). `K` is chosen **adaptively per attribute** to
  retain a target fraction of the signal energy (`--quality`), capped by `--max-coeffs`.
  Coefficients are stored as float16 (empirically lossless here — truncation dominates).

**Why DCT, not the plain DFT/Fourier:** the motion is *not periodic*, so a DFT sees a large
discontinuity at the last→first frame wrap and needs many coefficients (spectral leakage). The
DCT uses an even extension (no wrap discontinuity), so smooth non-periodic motion compresses
with far fewer coefficients — the same reason JPEG uses the DCT. Measured: at 16 coeffs the
DCT's position RMSE is ~0.0077 vs the DFT's ~0.0093 (see `compression_rate_distortion.png`,
right panel).

### Usage

```bash
# compress  (default quality 0.999 ≈ 10x at ~30 dB)
.venv/bin/python experiments/temporal_compress.py compress \
  splats/temporal/.../temporal_frames  splats/temporal/.../compressed \
  --quality 0.999 --max-coeffs 64

# decompress back into 00000.ply, 00001.ply, ...
.venv/bin/python experiments/temporal_compress.py decompress \
  splats/temporal/.../compressed  splats/temporal/.../frames_restored
```

The compressed directory holds `temporal.npz` (the coefficients) + a human-readable
`manifest.json` (properties, per-attribute mode/K, achieved ratio). Restored plys use the
standard nerfstudio gaussian-splat header and load in any 3DGS viewer / `ns-viewer`.

### Benchmark (123 frames × 4256 gaussians, 35.6 MB raw)

`quality` is the single quality knob; render PSNR is the compressed reconstruction vs the
original ply render (plots: `experiments/outputs/compression_{rate_distortion,per_property}.png`):

| `--quality` | ratio (on disk) | render PSNR vs original |
|---|---|---|
| 0.98 | **41.6×** | 25.1 dB |
| 0.99 | **25.7×** | 25.6 dB |
| 0.995 | **15.9×** | 26.4 dB |
| **0.999** *(default)* | **9.6×** | **30.1 dB** (per-frame mean 30.3, flat across the animation) |
| 0.9999 | 6.8× | 37.9 dB |

So the default already beats the 5× target by ~2× at ~30 dB; colours and normals are exactly
lossless, position error is ~0.9% of the scene span. Positions are cheap (x/y/z ≈ 15/19/7
coeffs), while the noisy opacity/rotation channels dominate the budget (hit the 64 cap) — lower
`--max-coeffs` for a smaller file, raise `--quality` for higher fidelity.

---

# Thesis benchmark suite (added 2026-07-02, rerun same day at 10k/3k)

All experiments for the thesis results chapter live in `experiments/` and write JSONs to
`experiments/outputs/`; the figure generators write **PNG (300 dpi)** figures straight into
`../artefacts/thesis/res/03_research/results/`. Trained runs for the chapter live under
`splats/thesis/<experiment-name>/temporal-splatfacto/thesis2/` (budgets N0=10000, Nt=3000 —
pass BOTH `--pipeline.model.{initial,tracking}-iterations` and
`--pipeline.datamanager.{initial,tracking}-iterations`).

**Background subtraction defaults were retuned** by a joint threshold/erosion sweep against
the GT silhouettes: `bg_subtract_threshold=0.06`, `bg_subtract_erode=2` (a new final-erosion
stage, N x 3x3 erosions after the dilation; the dilate+erode pair = closing + fringe
contraction). Cache dir default is now `images_bgsub_t06e2`. IoU 0.863/P 0.936/R 0.918 vs
0.824/0.933/0.878 for the old tau=0.1/no-erosion setting; pure-IoU optimum is (0.06, e=1)
at 0.884 — e=2 was chosen for the extra precision. Old-setting comparison run:
`full150_base` (tau=0.1, e=0, dir images_bgsub).

**Ground truth**: `output/black-bg-frames/static*` (room hidden; never seen by the pipeline).
The *illumination gap*: hiding the room removes bounce light, subject ~18% brighter in the
tuned segmented inputs; a pixel-perfect copy of the inputs scores only ~17.1 dB mean object
PSNR vs GT (15.3-19.8 per frame). Read all object-PSNR numbers against that reference.

| Script | What it does |
|---|---|
| `experiments/bgsub_benchmark.py` | Mask IoU/P/R + segmented-image PSNR vs GT; threshold sweep; **erosion sweep** (CPU) |
| `experiments/hull_init_benchmark.py` | Frame-0 fit from hull seed vs random seed (same budget / `--matched-budget 50000`) |
| `experiments/eval_temporal.py` | Offline eval of any run's `temporal_frames/`: object PSNR/SSIM, coverage, leak, extent, colour drift -> JSON |
| `experiments/compression_benchmark.py` | Quality sweep of `temporal_compress.py`: disk ratio + render PSNR, per-attribute K |
| `experiments/make_figures.py` | All matplotlib result figures as PNG (rerunnable; skips missing JSONs) |
| `experiments/make_qualitative.py` | PNG image figures: bg-sub montage, GT-vs-render montage, merged composite, trajectories |
| `experiments/seg_variants_benchmark.py` | Sweep of segmentation *variants* (mean/max diff, hysteresis ± shadow veto ± connectivity) vs GT masks (GPU) |
| `experiments/seg_full_eval.py` | Full 150x4 mask eval of the production vs hysteresis operating points (the thesis numbers) |
| `experiments/attribute_jitter.py` | Frame-to-frame mean abs change of position/opacity/scale/rotation from a `temporal_frames/` dir (CPU) |
| `experiments/make_improvement_figures.py` | Figures for the improvement experiments (seg variants/montage, snap margin, smoothness, end-to-end) |

Headline numbers (150 frames, 1080p, N0=10k/Nt=3k, tuned segmentation, RTX 6000 Ada):
71 min end-to-end, 11 561 gaussians (2.9 MB/frame, 430 MB), object PSNR 17.87 dB (above the
17.1 dB input reference on 149/150 frames), SSIM 0.854 flat over the sequence, leak 1.3e-3.
Old-segmentation flagship (`full150_base`): 16.13 dB / 0.783 — the retuned masks are worth
+1.7 dB / +0.07 SSIM end-to-end. Ablations (150 frames, half res, ~45 min each):
no snapping -> runaway extent 33.9; gating -> parked strays, worst SSIM 0.651 + leak 8.7e-3;
full-image loss -> background floods to opacity 0.99 by the last frame (PSNR decays to 10.5);
frozen opacity (no snap) -> SSIM 0.606; free colours -> f_dc drift 1.75 for no PSNR gain;
no velocity at Nt=3000 -> same end quality (velocity = efficiency at generous budgets; the
2026-07-06 rerun under the improved pipeline measures 23 vs 145 steps to 20 dB, 84% fewer,
plateau 29.3 vs 23.4 dB — and at the adopted Nt=300 budget velocity becomes load-bearing).
Hull-init (2026-07-06 rerun, hysteresis masks): hull seed 1631->18973 @41.7 dB, passes 30 dB
by ~step 950; equal-budget random collapses to 5.9 dB (686 prims); 50k random reaches
45.1 dB but needs 67k prims (~3.5x). Compression on this run: 430 MB -> 15.5 MB (27.8x) @ 26.2 dB with q=0.999 (colours
lossless via constant mode; opacity/scale/rot hit the 64-coeff cap — the 3k-step budget makes
non-positional channels noisier, hence lower ratio than short-budget runs). Background
splatfacto (dynamic1, 240 poses): 40.7 dB / 0.989 SSIM held-out (unchanged run `thesis`).

Remaining known limitation: residual camouflage holes vs the rug in early frames (reduced by
the tuned setting, not eliminated); strict snap test still flags ~half the cloud per frame on
imperfect masks (a majority test rho_s<1 would be gentler).

---

# Improvement campaign (2026-07-06)

Three mechanisms added on top of the v2 pipeline, each targeting one of its measured
limitations (all config-gated; defaults preserve the legacy behaviour). Runs under
`splats/thesis/<name>/temporal-splatfacto/imp/`; eval JSONs in `experiments/outputs/eval/`;
figures via `experiments/make_improvement_figures.py` (seg_variants, seg_montage_improved,
snap_margin, smoothness, budget_sweep, improved_summary — all PNG into the thesis res dir).

**Headline (150 frames, full res, N0=10k/Nt=3k):**

| run | mechanisms | N | SSIM | leak | notes |
|---|---|---|---|---|---|
| `full150` (baseline) | — | 11 561 | 0.854 | 1.3e-3 | v2 reference |
| `imp_seg` | hysteresis seg | 23 684 | 0.878 | 2.5e-5 | best solo; 2x primitives; camouflage window healed (first-20 SSIM 0.882 vs 0.830) |
| `imp_snapfield` | snap margin | 10 806 | 0.788 | 1.2e-3 | **hurts alone** — on noisy masks the blunt clamp was load-bearing |
| `imp_smooth` | smoothness λ=0.3 | 10 892 | 0.833 | 1.1e-3 | jitter -180x opacity; compression 27.8x->43.5x @ matched PSNR |
| `imp_full` | all three | 12 485 | **0.858** | 6.0e-5 | best overall; snaps drop to ~900/frame (10x fewer); 42.9x compression |

Key interaction: margin and hysteresis are complementary — the margin filters the *response*
to a noisy stray test, hysteresis cleans its *input*; alone the margin exposes mask noise
(SSIM 0.788), together they cut nominations 75%->22% of the cloud and teleports by 10x while
reaching the best SSIM. Per-frame snap diagnostics: `temporal_frames/temporal_stats.json`.

**Iteration-budget sweep (improved config, full 150 frames):** quality is FLAT across a
9.3x wall-clock range — `10k/3k` 72.8 min SSIM 0.858; `5k/1000` 23.7 min 0.860; `5k/500`
12.8 min 0.855; `3k/300` **7.8 min** 0.857 (all: coverage ~0.998, leak ~6e-5). Frame-0
converges by ~3–5k steps (38–39 dB; the extra 7k steps buy ~1 dB while densification
quadruples the cloud — see `experiments/outputs/hull_init/frame0_convergence_hyst.json`).

**Recommended default protocol for experiments** (adopted in the thesis):

```bash
--pipeline.model.initial-iterations 3000 \
--pipeline.model.tracking-iterations 300 \
--pipeline.datamanager.initial-iterations 3000 \
--pipeline.datamanager.tracking-iterations 300 \
--pipeline.datamanager.dataparser.bg-subtract-mode hysteresis \
--pipeline.datamanager.dataparser.bg-subtract-threshold 0.1 \
--pipeline.datamanager.dataparser.bg-subtract-dir images_bgsub_hyst \
--pipeline.model.snap-distance-margin 2.0 \
--pipeline.model.temporal-smoothness-lambda 0.3
```

~8 min for 150 frames at 1080p on one RTX 6000 Ada (vs ~73 min at the old 10k/3k), same
quality, and the smaller cloud (~6k gaussians) halves the per-frame files. Caveat: the small
Nt only works *because* velocity prediction is on — at Nt=300 a cold start cannot re-traverse
the inter-frame motion, so don't combine this budget with `--use-velocity-prediction False`.
