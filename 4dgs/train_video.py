"""Video reconstruction using Gaussian Splatting.

Reconstructs a video sequence where cameras are static but the scene changes
over time.  Uses full Gaussian Splatting (with densification) for the first
frame, then tracks subsequent frames by optimising without densification,
keeping the splat count fixed.

Improvement over naive per-frame tracking: linear velocity prediction
extrapolates gaussian positions from the two previous frames, giving a
better starting point and faster convergence.

Expected dataset layout
-----------------------
    dataset/
    ├── sparse/0/              # Standard COLMAP output
    │   ├── cameras.bin/txt
    │   ├── images.bin/txt     # References images like "cam01.jpg"
    │   └── points3D.bin/txt
    ├── images/                # First-frame images used by COLMAP
    │   ├── cam01.jpg
    │   └── cam02.jpg
    └── frames/                # Per-camera video frames  (--frames-dir)
        ├── cam01/             # Folder name = COLMAP image name without ext
        │   ├── 00000.jpg      # Sorted alphabetically = temporal order
        │   ├── 00001.jpg
        │   └── ...
        └── cam02/
            ├── 00000.jpg
            └── ...
"""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from random import randint
from typing import Annotated

import torch
import tyro
from tqdm import tqdm
from tyro.conf import OmitArgPrefixes

from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    expand_args,
    save_config,
)
from gaussian_renderer import render
from scene.dataset_readers import CameraInfo, readColmapSceneInfo, readNerfSyntheticInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import get_expon_lr_func, safe_state
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import SH2RGB

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

@dataclass
class VideoTrainConfig:
    model: Annotated[ModelParams, OmitArgPrefixes]
    opt: Annotated[OptimizationParams, OmitArgPrefixes]
    pipe: Annotated[PipelineParams, OmitArgPrefixes]

    frames_dir: str = "frames"
    """Directory under source_path with per-camera frame folders."""
    initial_iterations: int = 2000
    """Training iterations for the first frame (with densification)."""
    tracking_iterations: int = 500
    """Training iterations per subsequent frame (no densification)."""
    no_velocity: bool = False
    """Disable linear velocity prediction between frames."""
    start_frame: int = 0
    """Frame index to start from (for resuming)."""
    end_frame: int = -1
    """Frame index to end at, exclusive (-1 = all frames)."""
    prune_dark: float = 0.9
    """Prune gaussians darker than this fraction toward black (0-1). Negative = disabled."""
    prune_dark_interval: int = 100
    """Iterations between dark-gaussian prunes during the initial frame."""
    freeze_colors: bool = True
    """After the initial frame, keep SH colours (f_dc, f_rest) fixed during tracking."""
    quiet: bool = False


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def discover_frames(source_path: str, frames_dir: str, cam_infos: list):
    """Find per-camera frame directories and verify consistency.

    Returns
    -------
    frame_names : list[str]
        Sorted frame filenames shared by every camera.
    cam_frame_dirs : dict[str, str]
        Maps each camera's ``image_name`` to its frame directory path.
    """
    frames_root = os.path.join(source_path, frames_dir)
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Frames directory not found: {frames_root}")

    cam_frame_dirs: dict[str, str] = {}
    for cam in cam_infos:
        cam_name = os.path.splitext(cam.image_name)[0]
        cam_dir = os.path.join(frames_root, cam_name)
        if not os.path.isdir(cam_dir):
            raise FileNotFoundError(
                f"No frame directory for camera '{cam_name}': expected {cam_dir}"
            )
        cam_frame_dirs[cam.image_name] = cam_dir

    first_dir = next(iter(cam_frame_dirs.values()))
    frame_names = sorted(
        f for f in os.listdir(first_dir) if os.path.isfile(os.path.join(first_dir, f))
    )
    if not frame_names:
        raise ValueError(f"No frame files found in {first_dir}")

    for cam_name, cam_dir in cam_frame_dirs.items():
        cam_frames = sorted(
            f for f in os.listdir(cam_dir) if os.path.isfile(os.path.join(cam_dir, f))
        )
        if cam_frames != frame_names:
            raise ValueError(
                f"Frame mismatch for camera '{cam_name}': "
                f"expected {len(frame_names)} frames, found {len(cam_frames)}"
            )

    return frame_names, cam_frame_dirs


def build_frame_cam_infos(cam_infos, cam_frame_dirs, frame_name):
    """Create CameraInfo list for a specific frame by swapping image paths."""
    return [
        CameraInfo(
            uid=c.uid, R=c.R, T=c.T, FovY=c.FovY, FovX=c.FovX,
            depth_params=c.depth_params,
            image_path=os.path.join(cam_frame_dirs[c.image_name], frame_name),
            image_name=c.image_name,
            depth_path=c.depth_path,
            width=c.width, height=c.height, is_test=c.is_test,
        )
        for c in cam_infos
    ]


# ---------------------------------------------------------------------------
# Per-frame training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_dark_gaussians(gaussians: GaussianModel, prune_dark: float) -> tuple[int, int]:
    """Drop gaussians whose SH DC term encodes a colour darker than the threshold.

    Uses ``prune_points`` so optimizer state and auxiliary buffers stay consistent.
    Returns (n_before, n_after).
    """
    rgb = SH2RGB(gaussians._features_dc.squeeze(1))
    brightness = rgb.mean(dim=1)
    bright_threshold = 1.0 - prune_dark
    prune_mask = brightness < bright_threshold
    n_before = gaussians.num_points
    if prune_mask.any():
        gaussians.prune_points(prune_mask)
    return n_before, gaussians.num_points


def train_frame(
    gaussians: GaussianModel,
    cameras: list,
    opt: OptimizationParams,
    pipe: PipelineParams,
    background: torch.Tensor,
    num_iterations: int,
    densify: bool,
    cameras_extent: float,
    use_sparse_adam: bool,
    prune_dark: float = -1.0,
    prune_dark_interval: int = 100,
) -> float:
    """Run the optimisation loop for one frame.  Returns the final EMA loss."""
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=num_iterations
    )
    viewpoint_stack: list = []
    ema_loss = 0.0
    progress = tqdm(range(1, num_iterations + 1), desc="  iters", leave=False)

    for iteration in progress:
        gaussians.update_learning_rate(iteration)

        if densify and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = list(cameras)
        idx = randint(0, len(viewpoint_stack) - 1)
        cam = viewpoint_stack.pop(idx)

        render_pkg = render(cam, gaussians, pipe, background, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        viewspace_pts = render_pkg["viewspace_points"]
        vis_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if cam.alpha_mask is not None:
            image = image * cam.alpha_mask.cuda()

        gt = cam.original_image.cuda()
        Ll1 = l1_loss(image, gt)
        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(image.unsqueeze(0), gt.unsqueeze(0))
        else:
            ssim_val = ssim(image, gt)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        if depth_l1_weight(iteration) > 0 and cam.depth_reliable:
            inv_depth = render_pkg["depth"]
            mono = cam.invdepthmap.cuda()
            dmask = cam.depth_mask.cuda()
            loss = loss + depth_l1_weight(iteration) * torch.abs((inv_depth - mono) * dmask).mean()

        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress.set_postfix(loss=f"{ema_loss:.6f}", pts=gaussians.num_points)

            if densify and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis_filter] = torch.max(
                    gaussians.max_radii2D[vis_filter], radii[vis_filter]
                )
                gaussians.add_densification_stats(viewspace_pts, vis_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold, radii
                    )
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            if densify and prune_dark >= 0 and iteration % prune_dark_interval == 0:
                n_before, n_after = prune_dark_gaussians(gaussians, prune_dark)
                if n_before != n_after:
                    tqdm.write(f"  [iter {iteration}] dark prune: {n_before} -> {n_after}")

            if use_sparse_adam:
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)

    progress.close()
    return ema_loss


# ---------------------------------------------------------------------------
# Main video pipeline
# ---------------------------------------------------------------------------

def train_video(cfg: VideoTrainConfig):
    dataset = cfg.model
    opt = cfg.opt
    pipe = cfg.pipe
    use_velocity = not cfg.no_velocity

    # ---- output directory ----
    if not dataset.model_path:
        dataset.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])
    print(f"Output folder: {dataset.model_path}")
    os.makedirs(dataset.model_path, exist_ok=True)
    save_config(dataset, os.path.join(dataset.model_path, "cfg_args"))

    # ---- load scene info ----
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = readColmapSceneInfo(
            dataset.source_path, dataset.images, dataset.depths, eval=False, train_test_exp=False
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        scene_info = readNerfSyntheticInfo(
            dataset.source_path, dataset.white_background, dataset.depths, eval=False
        )
    else:
        sys.exit("Could not recognise scene type (expected sparse/ or transforms_train.json)")

    cameras_extent = scene_info.nerf_normalization["radius"]
    all_cam_infos = scene_info.train_cameras + scene_info.test_cameras
    is_synthetic = scene_info.is_nerf_synthetic

    # ---- discover video frames ----
    frame_names, cam_frame_dirs = discover_frames(dataset.source_path, cfg.frames_dir, all_cam_infos)
    num_frames = len(frame_names)
    start = cfg.start_frame
    end = cfg.end_frame if cfg.end_frame >= 0 else num_frames
    end = min(end, num_frames)

    print(f"Cameras:  {len(all_cam_infos)}")
    print(f"Frames:   {num_frames}  (processing {start}..{end - 1})")
    print(f"Initial iterations:  {cfg.initial_iterations}")
    print(f"Tracking iterations: {cfg.tracking_iterations}")
    print(f"Velocity prediction: {'on' if use_velocity else 'off'}")

    # ---- setup ----
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    prev_xyz: torch.Tensor | None = None
    prev_prev_xyz: torch.Tensor | None = None

    # ---- resume from a later frame ----
    if start > 0:
        prev_ply = os.path.join(dataset.model_path, "frames", f"{start - 1:05d}.ply")
        if not os.path.isfile(prev_ply):
            sys.exit(f"Cannot resume from frame {start}: {prev_ply} not found")
        print(f"Resuming: loading PLY from frame {start - 1}")
        gaussians.load_ply(prev_ply)
        gaussians.exposure_mapping = {c.image_name: i for i, c in enumerate(all_cam_infos)}
        gaussians.pretrained_exposures = None
        gaussians._exposure = torch.nn.Parameter(
            torch.eye(3, 4, device="cuda")[None].repeat(len(all_cam_infos), 1, 1).requires_grad_(True)
        )
        gaussians.spatial_lr_scale = cameras_extent
        prev_xyz = gaussians.get_xyz.detach().clone()

        if start > 1 and use_velocity:
            pp_ply = os.path.join(dataset.model_path, "frames", f"{start - 2:05d}.ply")
            if os.path.isfile(pp_ply):
                import numpy as np
                from plyfile import PlyData
                ply = PlyData.read(pp_ply)
                v = ply.elements[0]
                pp_xyz = np.stack((np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])), axis=1)
                prev_prev_xyz = torch.tensor(pp_xyz, dtype=torch.float, device="cuda")

    # ---- frame loop ----
    for frame_idx in range(start, end):
        frame_name = frame_names[frame_idx]
        print(f"\n{'=' * 60}")
        print(f"Frame {frame_idx}/{num_frames - 1}  ({frame_name})")
        print(f"{'=' * 60}")

        frame_cam_infos = build_frame_cam_infos(all_cam_infos, cam_frame_dirs, frame_name)
        cameras = cameraList_from_camInfos(frame_cam_infos, 1.0, dataset, is_synthetic, False)

        if frame_idx == start and start == 0:
            # --- Initial reconstruction ---
            gaussians.create_from_pcd(scene_info.point_cloud, frame_cam_infos, cameras_extent)
            gaussians.training_setup(opt)

            print(f"Initial reconstruction: {cfg.initial_iterations} iterations")
            final_loss = train_frame(
                gaussians, cameras, opt, pipe, background,
                num_iterations=cfg.initial_iterations, densify=True,
                cameras_extent=cameras_extent, use_sparse_adam=use_sparse_adam,
                prune_dark=cfg.prune_dark, prune_dark_interval=cfg.prune_dark_interval,
            )
            if cfg.prune_dark >= 0:
                n_before, n_after = prune_dark_gaussians(gaussians, cfg.prune_dark)
                print(f"Final dark prune: {n_before} -> {n_after} "
                      f"(brightness < {1.0 - cfg.prune_dark:.3f})")
        else:
            # --- Tracking ---
            if use_velocity and prev_prev_xyz is not None:
                velocity = prev_xyz - prev_prev_xyz
                gaussians._xyz.data.add_(velocity)
                print(f"Velocity prediction (mean displacement: {velocity.norm(dim=1).mean():.6f})")

            gaussians.training_setup(opt)
            gaussians.active_sh_degree = gaussians.max_sh_degree

            if cfg.freeze_colors:
                for group in gaussians.optimizer.param_groups:
                    if group["name"] in ("f_dc", "f_rest"):
                        group["lr"] = 0.0
                        for p in group["params"]:
                            p.requires_grad_(False)

            print(f"Tracking: {cfg.tracking_iterations} iterations, {gaussians.num_points} gaussians"
                  + (" (colors frozen)" if cfg.freeze_colors else ""))
            final_loss = train_frame(
                gaussians, cameras, opt, pipe, background,
                num_iterations=cfg.tracking_iterations, densify=False,
                cameras_extent=cameras_extent, use_sparse_adam=use_sparse_adam,
            )

        prev_prev_xyz = prev_xyz
        prev_xyz = gaussians.get_xyz.detach().clone()

        frames_dir = os.path.join(dataset.model_path, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        ply_path = os.path.join(frames_dir, f"{frame_idx:05d}.ply")
        gaussians.save_ply(ply_path)
        print(f"Saved: {ply_path}  ({gaussians.num_points} gaussians, loss={final_loss:.6f})")

    print(f"\nVideo reconstruction complete. Frames {start}..{end - 1} saved to {dataset.model_path}/frames/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = tyro.cli(VideoTrainConfig, args=expand_args())
    safe_state(cfg.quiet)
    train_video(cfg)
