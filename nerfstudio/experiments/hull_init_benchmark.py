#!/usr/bin/env python
"""Experiment: visual-hull seeding vs random initialization for the frame-0 fit.

Fits frame 0 twice from scratch — once seeded from the carved visual hull, once from a
random cloud in the camera bounding box (splatfacto's seed-free default) with the *same*
initial primitive budget — running the full splatfacto optimizer with densification, and
records the object-region PSNR and the gaussian count as training progresses.

Run::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/hull_init_benchmark.py \
        --data ../output/dataset
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tyro

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Config:
    data: Path = Path("../output/dataset")
    initial_iterations: int = 3000
    camera_res_scale_factor: float = 0.5
    eval_every: int = 25
    fg_threshold: float = 0.05
    matched_budget: int = -1
    """Random-init budget; -1 = match the hull's seed count."""
    output_dir: Path = Path("experiments/outputs/hull_init")
    device: str = "cuda"
    seed: int = 0


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (pred - gt)[mask]
    mse = (diff**2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def build_pipeline(cfg: Config, use_hull: bool, num_random: int):
    import copy

    trainer_cfg = copy.deepcopy(all_methods["temporal-splatfacto"])
    trainer_cfg.pipeline.datamanager.data = cfg.data
    trainer_cfg.pipeline.datamanager.dataparser.data = cfg.data
    trainer_cfg.pipeline.datamanager.dataparser.max_frames = 1
    trainer_cfg.pipeline.datamanager.dataparser.init_visual_hull = use_hull
    trainer_cfg.pipeline.datamanager.camera_res_scale_factor = cfg.camera_res_scale_factor
    trainer_cfg.pipeline.model.initial_iterations = cfg.initial_iterations
    trainer_cfg.pipeline.model.tracking_iterations = 1
    trainer_cfg.pipeline.model.num_random = num_random
    trainer_cfg.pipeline.model.background_color = "black"
    pipeline = trainer_cfg.pipeline.setup(device=cfg.device, test_mode="val", world_size=1, local_rank=0)
    return pipeline, trainer_cfg


def fit_frame0(cfg: Config, use_hull: bool, num_random: int) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    pipeline, trainer_cfg = build_pipeline(cfg, use_hull, num_random)
    model = pipeline.model
    dm = pipeline.datamanager

    from nerfstudio.utils.temporal_schedule import TemporalSchedule

    schedule = TemporalSchedule(
        initial_iterations=cfg.initial_iterations, tracking_iterations=1, num_frames=dm.num_frames
    )
    model.set_schedule(schedule)

    views0 = dm.frame_to_indices[0]
    imgs = [dm.cached_train[i]["image"].to(cfg.device) for i in views0]
    cams = [dm.train_cameras[i : i + 1].to(cfg.device) for i in views0]
    bg = torch.zeros(3, device=cfg.device)
    gts = [model.composite_with_background(model.get_gt_img(img), bg) for img in imgs]
    masks = [g[..., :3].mean(-1) > cfg.fg_threshold for g in gts]

    opt = Optimizers(trainer_cfg.optimizers.copy(), model.get_param_groups())
    model.train()
    rng = np.random.RandomState(cfg.seed)

    curve_steps: List[int] = []
    curve_psnr: List[float] = []
    curve_count: List[int] = []
    init_count = model.num_points
    CONSOLE.log(f"[hull-init] arm={'hull' if use_hull else 'random'}: {init_count} initial gaussians")

    for step in range(cfg.initial_iterations):
        model.step_cb(opt, step)  # sets step/optimizers/LR, drives the schedule
        opt.zero_grad_all()
        i = int(rng.randint(len(cams)))  # one camera per step, like the real datamanager
        out = model.get_outputs(cams[i])
        loss = sum(model.get_loss_dict(out, {"image": imgs[i]}).values())
        loss.backward()
        opt.optimizer_step_all()
        model.step_post_backward(step)  # densification / pruning

        if step % cfg.eval_every == 0 or step == cfg.initial_iterations - 1:
            model.eval()
            with torch.no_grad():
                psnrs = [
                    masked_psnr(model.get_outputs(c)["rgb"], g, m) for c, g, m in zip(cams, gts, masks)
                ]
            model.train()
            curve_steps.append(step)
            curve_psnr.append(float(np.mean(psnrs)))
            curve_count.append(model.num_points)
            if step % (cfg.eval_every * 10) == 0:
                CONSOLE.log(
                    f"[hull-init] {'hull' if use_hull else 'random'} step {step}: "
                    f"objPSNR={curve_psnr[-1]:.2f}, {curve_count[-1]} gaussians"
                )

    result = {
        "init_count": init_count,
        "final_count": curve_count[-1],
        "steps": curve_steps,
        "object_psnr": curve_psnr,
        "num_gaussians": curve_count,
        "final_psnr": curve_psnr[-1],
    }
    del pipeline, model, dm, opt
    torch.cuda.empty_cache()
    return result


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    hull = fit_frame0(cfg, use_hull=True, num_random=1000)  # num_random unused when seeded
    budget = cfg.matched_budget if cfg.matched_budget > 0 else hull["init_count"]
    rand = fit_frame0(cfg, use_hull=False, num_random=budget)
    out = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "hull": hull,
        "random": rand,
    }
    (cfg.output_dir / "hull_init_benchmark.json").write_text(json.dumps(out, indent=2))
    CONSOLE.print(
        f"[green]done[/green] hull: init={hull['init_count']} final PSNR={hull['final_psnr']:.2f} | "
        f"random: init={rand['init_count']} final PSNR={rand['final_psnr']:.2f}"
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
