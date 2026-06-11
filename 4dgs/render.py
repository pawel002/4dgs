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
from scene import Scene, GaussianModel
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False


@dataclass
class RenderConfig:
    model: Annotated[ModelParams, OmitArgPrefixes]
    pipe: Annotated[PipelineParams, OmitArgPrefixes]
    iteration: int = -1
    skip_train: bool = False
    skip_test: bool = False
    quiet: bool = False


def _load_defaults() -> RenderConfig:
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
                    return RenderConfig(model=saved)
                except Exception:
                    pass
    return RenderConfig()


def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background,
                           use_trained_exp=train_test_exp,
                           separate_sh=separate_sh)["render"]
        gt = view.original_image[:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))


def render_sets(cfg: RenderConfig):
    dataset = cfg.model
    pipeline = cfg.pipe

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=cfg.iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not cfg.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE)

        if not cfg.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline,
                       background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE)


if __name__ == "__main__":
    defaults = _load_defaults()
    cfg = tyro.cli(RenderConfig, default=defaults, args=expand_args())

    print("Rendering " + cfg.model.model_path)
    safe_state(cfg.quiet)
    render_sets(cfg)
