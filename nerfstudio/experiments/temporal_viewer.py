#!/usr/bin/env python
"""Temporal (4D) Gaussian-Splatting playback viewer — viser rendering engine.

After training ``temporal-splatfacto`` you get one ``.ply`` per video frame in
``temporal_frames/`` (the moving *foreground*), and optionally a single static
``.ply`` exported from ``splatfacto`` (the *background*). This tool loads them
into a `viser <https://viser.studio>`_ web viewer and plays the animation:

* **Static background + dynamic foreground composited together.** The background
  ply is added once (always visible); the per-frame foreground plys are added as
  one scene node each, and only the current frame's node is left visible — so the
  background stays put while the foreground animates over it.
* **Playback controls** (in the viewer's right-hand panel):
    - ``▶ Play / ⏸ Pause`` — start/stop automatic playback.
    - ``Loop`` — when on, playback wraps from the last frame back to the first;
      when off, it stops on the last frame.
    - ``FPS`` — playback framerate (frames advanced per second).
    - ``Frame`` — a slider you can drag to scrub to any frame while paused (or
      playing); ``◀`` / ``▶`` buttons step one frame at a time.

How frame switching is instant
------------------------------
Every frame is uploaded to the browser **once** at startup as its own Gaussian-
splat scene node (all hidden except frame 0). Stepping/playing only flips each
node's ``visible`` flag — no per-frame re-upload — so playback is smooth even at
high FPS. The 4DGS output keeps the gaussian count constant (~a few thousand)
across frames, so holding all frames resident is cheap.

Usage
-----
    # point it at the temporal_frames/ directory (or the run dir above it):
    python experiments/temporal_viewer.py \
        splats/temporal/temporal/temporal-splatfacto/<timestamp>/temporal_frames

    # composite a static background exported from splatfacto:
    python experiments/temporal_viewer.py <frames_dir> \
        --background splats/dynamic/export/splat.ply --fps 30

Then open http://localhost:7007 (forward the port over SSH if remote).
"""

from __future__ import annotations

import argparse
import glob
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import viser

# Reuse the all-float32 binary PLY reader from the compression tool.
from temporal_compress import read_ply

SH_C0 = 0.28209479177387814  # 0th-order spherical-harmonic basis (DC term -> base colour)


# ----------------------------------------------------------------------- ply -> gaussians
def _col(props: List[str], data: np.ndarray, name: str) -> np.ndarray:
    return data[:, props.index(name)]


def ply_to_gaussians(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a nerfstudio gaussian-splat ply into viser's expected arrays.

    Returns ``(centers[N,3], covariances[N,3,3], rgbs[N,3] in 0..1, opacities[N,1] in 0..1)``,
    applying the standard activations: opacity = sigmoid(stored), scale = exp(stored),
    quaternion (wxyz) normalised, colour = 0.5 + SH_C0 * f_dc.
    """
    props, data = read_ply(path)

    centers = np.stack([_col(props, data, "x"), _col(props, data, "y"), _col(props, data, "z")], axis=1)

    f_dc = np.stack([_col(props, data, f"f_dc_{i}") for i in range(3)], axis=1)
    rgbs = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    opacities = (1.0 / (1.0 + np.exp(-_col(props, data, "opacity"))))[:, None]

    scales = np.exp(np.stack([_col(props, data, f"scale_{i}") for i in range(3)], axis=1))
    quats = np.stack([_col(props, data, f"rot_{i}") for i in range(4)], axis=1)  # wxyz
    covariances = _scales_quats_to_covariances(scales, quats)

    return centers.astype(np.float32), covariances.astype(np.float32), rgbs.astype(np.float32), opacities.astype(np.float32)


def _scales_quats_to_covariances(scales: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Build per-gaussian 3x3 covariance = R diag(s^2) R^T from scales (N,3) and wxyz quats (N,4)."""
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    rot = np.empty((n, 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    # M = R * s (scale the columns of R), cov = M M^T.
    m = rot * scales[:, None, :]
    return m @ m.transpose(0, 2, 1)


# ----------------------------------------------------------------------- frame discovery
def find_frame_plys(frames_dir: str, max_frames: int) -> List[str]:
    """Return sorted per-frame plys; if ``frames_dir`` has none, look in temporal_frames/ under it."""
    d = Path(frames_dir)
    files = sorted(glob.glob(str(d / "*.ply")))
    if not files and (d / "temporal_frames").is_dir():
        files = sorted(glob.glob(str(d / "temporal_frames" / "*.ply")))
    if not files:
        raise SystemExit(f"No .ply frames found in {frames_dir} (or its temporal_frames/ subdir).")
    if max_frames > 0:
        files = files[:max_frames]
    return files


# ----------------------------------------------------------------------- viewer
def run_viewer(
    frames_dir: str,
    background: Optional[str],
    fps: float,
    port: int,
    max_frames: int,
) -> None:
    files = find_frame_plys(frames_dir, max_frames)
    num_frames = len(files)
    print(f"[viewer] loading {num_frames} frames from {frames_dir} ...")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")  # temporal plys are stored z-up

    # Static background (optional) — added once, always visible, behind the animation.
    if background is not None:
        c, cov, rgb, op = ply_to_gaussians(background)
        server.scene.add_gaussian_splats("/background", centers=c, covariances=cov, rgbs=rgb, opacities=op)
        print(f"[viewer] background: {background} ({c.shape[0]} gaussians)")

    # Dynamic foreground — every frame uploaded once as its own node, hidden except frame 0.
    handles = []
    for i, fpath in enumerate(files):
        c, cov, rgb, op = ply_to_gaussians(fpath)
        h = server.scene.add_gaussian_splats(
            f"/frames/{i:05d}", centers=c, covariances=cov, rgbs=rgb, opacities=op, visible=(i == 0)
        )
        handles.append(h)
        if (i + 1) % 25 == 0 or i + 1 == num_frames:
            print(f"[viewer]   uploaded {i + 1}/{num_frames} frames")

    # ---- state shared between the GUI callbacks and the playback thread.
    state = {"frame": 0, "playing": False, "suppress": False}
    lock = threading.Lock()

    # ---- GUI.
    server.gui.add_markdown("### Temporal playback")
    play_button = server.gui.add_button("▶ Play")
    loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=int(fps))
    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    with server.gui.add_folder("Step"):
        prev_button = server.gui.add_button("◀ Prev")
        next_button = server.gui.add_button("Next ▶")
    frame_label = server.gui.add_text("Showing", initial_value=f"frame 1 / {num_frames}", disabled=True)

    def show_frame(idx: int) -> None:
        """Flip visibility to ``idx`` and sync the GUI (the single place frames change)."""
        idx = int(idx) % num_frames
        with lock:
            prev = state["frame"]
            if prev != idx:
                handles[prev].visible = False
            handles[idx].visible = True
            state["frame"] = idx
            # Update the slider without re-triggering its callback.
            state["suppress"] = True
            frame_slider.value = idx
            state["suppress"] = False
            frame_label.value = f"frame {idx + 1} / {num_frames}"

    def set_playing(playing: bool) -> None:
        state["playing"] = playing
        play_button.label = "⏸ Pause" if playing else "▶ Play"

    @play_button.on_click
    def _(_) -> None:
        playing = not state["playing"]
        # If starting playback from the last frame (and not looping), restart at 0.
        if playing and not loop_checkbox.value and state["frame"] >= num_frames - 1:
            show_frame(0)
        set_playing(playing)

    @frame_slider.on_update
    def _(_) -> None:
        if state["suppress"]:
            return
        set_playing(False)  # manual scrubbing pauses playback
        show_frame(frame_slider.value)

    @prev_button.on_click
    def _(_) -> None:
        set_playing(False)
        show_frame(state["frame"] - 1)

    @next_button.on_click
    def _(_) -> None:
        set_playing(False)
        show_frame(state["frame"] + 1)

    # ---- playback thread: advances frames at the chosen FPS while playing.
    def playback_loop() -> None:
        while True:
            if state["playing"]:
                cur = state["frame"]
                if cur >= num_frames - 1:
                    if loop_checkbox.value:
                        show_frame(0)
                    else:
                        set_playing(False)  # stop on the last frame
                else:
                    show_frame(cur + 1)
            time.sleep(1.0 / max(1, fps_slider.value))

    threading.Thread(target=playback_loop, daemon=True).start()

    print(f"[viewer] ready — open http://localhost:{port}  (forward the port over SSH if remote)")
    print("[viewer] Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viewer] bye.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "frames_dir",
        help="Directory of per-frame .ply files (temporal_frames/), or a run dir that contains it.",
    )
    ap.add_argument("--background", default=None, help="Optional static background .ply (e.g. splatfacto export).")
    ap.add_argument("--fps", type=float, default=30.0, help="Initial playback framerate (adjustable in the GUI).")
    ap.add_argument("--port", type=int, default=7007, help="viser websocket port.")
    ap.add_argument("--max-frames", type=int, default=-1, help="Cap the number of frames loaded (-1 = all).")
    a = ap.parse_args()
    run_viewer(a.frames_dir, a.background, a.fps, a.port, a.max_frames)


if __name__ == "__main__":
    main()
