# Video Gaussian Splatting

## What changed

### Argument parsing: argparse → tyro

All scripts now use [**tyro**](https://github.com/brentyi/tyro) for argument parsing. Parameters are defined as plain `@dataclass` classes in `arguments/__init__.py` — no metaclass hacks, no `extract()` methods.

**tyro** generates the CLI automatically from type hints. Flags use dashes (`--source-path`) but underscores also work (`--source_path`). The legacy short flags (`-s`, `-m`, `-i`, `-d`, `-r`, `-w`) are preserved.

Config is now saved as JSON (the old `Namespace(...)` format is still loadable for backward compat).

Install: `pip install tyro` (also added to `environment.yml`).

### New files
- **`train_video.py`** — Video reconstruction. Trains frame 0 with full densification, then tracks subsequent frames with a fixed splat count.
- **`render_video.py`** — Renders all saved video frames from their PLY files.

### Modified files
- **`arguments/__init__.py`** — Rewritten as pure dataclasses. Added `save_config` / `load_config` (JSON) and `expand_args` (short-flag compat).
- **`train.py`** — Uses `tyro.cli(TrainConfig)`. Same training logic.
- **`render.py`** — Uses `tyro.cli(RenderConfig)` with auto-loading of saved config from `--model-path`.
- **`scene/gaussian_model.py`** — Fixed `load_ply` bug (SLICE=2), removed `tmp_radii` hack, added `num_points` property.
- **`gaussian_renderer/__init__.py`** — Removed debug print statements.
- **`scene/__init__.py`** — Cleaned up.
- **`environment.yml`** — Added `tyro`.

### Algorithm: velocity prediction
For frames beyond the second, gaussian positions are extrapolated linearly:

```
predicted_pos[k] = pos[k-1] + (pos[k-1] - pos[k-2])
```

This gives a better starting point when the scene has consistent motion. Disable with `--no-velocity`.

---

## Standard training (single scene)

```bash
python train.py --source-path /data/scene --model-path /output/scene

# Short flags still work:
python train.py -s /data/scene -m /output/scene
```

See all options: `python train.py --help`

---

## Video training

### Dataset layout

```
dataset/
├── sparse/0/              # Standard COLMAP output
│   ├── cameras.bin        # Camera intrinsics
│   ├── images.bin         # Camera extrinsics (references e.g. "cam01.jpg")
│   └── points3D.bin       # Initial point cloud
├── images/                # First-frame images (the ones COLMAP was run on)
│   ├── cam01.jpg
│   ├── cam02.jpg
│   └── ...
└── frames/                # Video frames per camera
    ├── cam01/             # Folder name = COLMAP image name without extension
    │   ├── 00000.jpg      # Frames sorted alphabetically = temporal order
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   └── ...
    └── cam02/
        ├── 00000.jpg
        ├── 00001.jpg
        └── ...
```

**Rules:**
- Every camera folder must contain the same set of frame filenames.
- Frame filenames are sorted alphabetically to determine temporal order.
- Camera folder names = COLMAP image names without extension (e.g. `cam01.jpg` → `frames/cam01/`).

### Basic usage

```bash
python train_video.py -s path/to/dataset -m path/to/output \
    --frames-dir frames \
    --initial-iterations 7000 \
    --tracking-iterations 500
```

### All video arguments

| Argument | Default | Description |
|---|---|---|
| `-s` / `--source-path` | (required) | Path to dataset root |
| `-m` / `--model-path` | auto-generated | Output directory |
| `--frames-dir` | `frames` | Directory under source containing per-camera frame folders |
| `--initial-iterations` | `7000` | Training iterations for frame 0 (with densification) |
| `--tracking-iterations` | `500` | Training iterations for subsequent frames (no densification) |
| `--no-velocity` | off | Disable velocity prediction |
| `--start-frame` | `0` | Frame index to start from (for resuming) |
| `--end-frame` | `-1` | Frame index to stop at, exclusive (`-1` = all) |
| `--quiet` | off | Suppress output |

All standard optimisation flags also work (`--iterations`, `--position-lr-init`, `--lambda-dssim`, etc.).

### Resuming

```bash
python train_video.py -s dataset/ -m output/ --start-frame 50
```

Loads the PLY from frame 49 and continues. Velocity prediction also loads frame 48.

### Output structure

```
output/
├── cfg_args               # Saved config (JSON)
└── frames/
    ├── 00000/
    │   └── point_cloud.ply
    ├── 00001/
    │   └── point_cloud.ply
    └── ...
```

---

## Rendering video

```bash
python render_video.py -s path/to/dataset -m path/to/output --frames-dir frames
```

Add `--skip-gt` to skip saving ground-truth images.

Output:

```
output/
├── video_renders/
│   ├── cam01/
│   │   ├── 00000.png
│   │   └── ...
│   └── cam02/
│       └── ...
└── video_gt/
    └── ...
```

---

## Quick start

```bash
# 1. Prepare dataset (run COLMAP on first-frame images, organize frames/)

# 2. Train
python train_video.py -s my_scene/ -m output/video/ \
    --initial-iterations 7000 \
    --tracking-iterations 500

# 3. Render
python render_video.py -s my_scene/ -m output/video/

# 4. Make a video
ffmpeg -framerate 30 -i output/video/video_renders/cam01/%05d.png \
    -c:v libx264 -pix_fmt yuv420p video_cam01.mp4
```
