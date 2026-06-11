"""Render all frames of a video reconstructed by train_video.py."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Annotated

import torch
import torchvision
import tyro
from tqdm import tqdm
from tyro.conf import OmitArgPrefixes

from arguments import ModelParams, PipelineParams, expand_args, load_config
from gaussian_renderer import render
from scene.dataset_readers import CameraInfo, readColmapSceneInfo, readNerfSyntheticInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


@dataclass
class VideoRenderConfig:
    model: Annotated[ModelParams, OmitArgPrefixes]
    pipe: Annotated[PipelineParams, OmitArgPrefixes]
    frames_dir: str = "frames"
    """Source frames directory name (for GT images)."""
    skip_gt: bool = False
    """Skip saving ground-truth images."""
    quiet: bool = False


def _load_defaults() -> VideoRenderConfig:
    """Scan argv for --model-path, load saved config as defaults."""
    argv = expand_args()
    model_path = ""
    for i, tok in enumerate(argv):
        if tok in ("--model-path", "--model_path") and i + 1 < len(argv):
            model_path = argv[i + 1]
            break
    if model_path:
        for name in ("cfg_args", "cfg_args.json"):
            path = os.path.join(model_path, name)
            if os.path.isfile(path):
                try:
                    saved = load_config(path)
                    saved.model_path = model_path
                    return VideoRenderConfig(model=saved)
                except Exception:
                    pass
    return VideoRenderConfig()


def discover_saved_frames(model_path: str) -> list[int]:
    frames_dir = os.path.join(model_path, "frames")
    if not os.path.isdir(frames_dir):
        sys.exit(f"No frames directory found at {frames_dir}")
    indices: list[int] = []
    for name in sorted(os.listdir(frames_dir)):
        ply = os.path.join(frames_dir, name, "point_cloud.ply")
        if os.path.isfile(ply):
            try:
                indices.append(int(name))
            except ValueError:
                continue
    return indices


def discover_source_frames(source_path, frames_dir_name, cam_infos):
    frames_root = os.path.join(source_path, frames_dir_name)
    if not os.path.isdir(frames_root):
        return None, None
    cam_frame_dirs: dict[str, str] = {}
    for cam in cam_infos:
        cam_name = os.path.splitext(cam.image_name)[0]
        cam_dir = os.path.join(frames_root, cam_name)
        if os.path.isdir(cam_dir):
            cam_frame_dirs[cam.image_name] = cam_dir
    if not cam_frame_dirs:
        return None, None
    first_dir = next(iter(cam_frame_dirs.values()))
    frame_names = sorted(
        f for f in os.listdir(first_dir) if os.path.isfile(os.path.join(first_dir, f))
    )
    return frame_names, cam_frame_dirs


def build_frame_cam_infos(cam_infos, cam_frame_dirs, frame_name):
    return [
        CameraInfo(
            uid=c.uid, R=c.R, T=c.T, FovY=c.FovY, FovX=c.FovX,
            depth_params=c.depth_params,
            image_path=os.path.join(cam_frame_dirs[c.image_name], frame_name),
            image_name=c.image_name,
            depth_path=c.depth_path, width=c.width, height=c.height, is_test=c.is_test,
        )
        for c in cam_infos if c.image_name in cam_frame_dirs
    ]


def render_video(cfg: VideoRenderConfig):
    dataset = cfg.model
    pipeline = cfg.pipe

    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = readColmapSceneInfo(
            dataset.source_path, dataset.images, dataset.depths, eval=False, train_test_exp=False
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        scene_info = readNerfSyntheticInfo(
            dataset.source_path, dataset.white_background, dataset.depths, eval=False
        )
    else:
        sys.exit("Could not recognise scene type")

    all_cam_infos = scene_info.train_cameras + scene_info.test_cameras
    is_synthetic = scene_info.is_nerf_synthetic

    frame_indices = discover_saved_frames(dataset.model_path)
    if not frame_indices:
        sys.exit("No saved frame PLYs found")
    print(f"Found {len(frame_indices)} saved frames")

    frame_names, cam_frame_dirs = discover_source_frames(
        dataset.source_path, cfg.frames_dir, all_cam_infos
    )
    has_gt = frame_names is not None and cam_frame_dirs is not None and not cfg.skip_gt

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(dataset.sh_degree)

    render_base = os.path.join(dataset.model_path, "video_renders")
    gt_base = os.path.join(dataset.model_path, "video_gt") if has_gt else None

    with torch.no_grad():
        for frame_idx in tqdm(frame_indices, desc="Frames"):
            ply_path = os.path.join(dataset.model_path, "frames", f"{frame_idx:05d}", "point_cloud.ply")
            gaussians.load_ply(ply_path)

            if has_gt and frame_idx < len(frame_names):
                frame_cam_infos = build_frame_cam_infos(all_cam_infos, cam_frame_dirs, frame_names[frame_idx])
            else:
                frame_cam_infos = all_cam_infos
            cameras = cameraList_from_camInfos(frame_cam_infos, 1.0, dataset, is_synthetic, False)

            for cam in cameras:
                cam_name = os.path.splitext(cam.image_name)[0]
                rendering = torch.clamp(
                    render(cam, gaussians, pipeline, background, separate_sh=SPARSE_ADAM_AVAILABLE)["render"],
                    0.0, 1.0,
                )

                render_dir = os.path.join(render_base, cam_name)
                os.makedirs(render_dir, exist_ok=True)
                torchvision.utils.save_image(rendering, os.path.join(render_dir, f"{frame_idx:05d}.png"))

                if gt_base is not None:
                    gt_dir = os.path.join(gt_base, cam_name)
                    os.makedirs(gt_dir, exist_ok=True)
                    torchvision.utils.save_image(
                        cam.original_image[:3], os.path.join(gt_dir, f"{frame_idx:05d}.png")
                    )

    print(f"Renders saved to: {render_base}")
    if gt_base:
        print(f"Ground truth saved to: {gt_base}")


if __name__ == "__main__":
    defaults = _load_defaults()
    cfg = tyro.cli(VideoRenderConfig, default=defaults, args=expand_args())
    safe_state(cfg.quiet)
    render_video(cfg)
