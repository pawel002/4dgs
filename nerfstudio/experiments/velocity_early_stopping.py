#!/usr/bin/env python
"""Experiment: does velocity prediction reduce the steps needed to fit each frame?

For the temporal (4D) gaussian splatting method we measure, **per frame**, how many
optimization steps are needed for the per-frame fit to *converge* — and compare that with
velocity prediction on vs off.

Early-stopping metric
---------------------
Convergence is decided by :class:`EarlyStopper` on the **mean PSNR over the frame's static
cameras** (a standard image-reconstruction quality metric). A frame is converged once its
PSNR stops improving by more than ``min_delta`` dB for ``patience`` consecutive steps (a
plateau), or reaches ``target_psnr`` if given. The *step at which this happens* is the
frame's "steps to convergence". Both conditions use the identical criterion, so they are
compared at the same quality.

Protocol
--------
1. Build the temporal pipeline (gaussians seeded from the carved visual hull, black
   background) and fit **frame 0** once; snapshot that state.
2. For ``use_velocity in {False, True}``: restore the frame-0 snapshot and track frames
   1..N sequentially, each frame optimized (4 views/step, colours frozen, optimizer reset)
   until the early stopper fires. Record the convergence step per frame. With velocity on,
   each new frame is warm-started by extrapolating ``x_t += x_{t-1} - x_{t-2}``.
3. Plot the distribution of convergence steps for both conditions as overlaid histograms,
   with each condition's mean marked, and save a summary + the figure.

Run::

    CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python experiments/velocity_early_stopping.py --data ../output --num-frames 40
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.temporal_schedule import EarlyStopper


@dataclass
class ExperimentConfig:
    data: Path = Path("../output")
    """Parent directory containing the static* camera datasets."""
    num_frames: int = 40
    """Number of temporal frames to use (frame 0 + the tracking frames after it)."""
    initial_iterations: int = 1000
    """Steps used to fit frame 0 (shared identical starting point for both conditions)."""
    max_track_steps: int = 400
    """Hard cap on per-frame tracking steps (a frame that never converges is censored here)."""
    patience: int = 25
    """Early-stop patience: steps without a PSNR improvement before declaring convergence."""
    min_delta: float = 0.05
    """Early-stop min PSNR (dB) improvement that counts as progress."""
    fg_threshold: float = 0.05
    """Brightness (0-1) above which a GT pixel is object foreground. The early-stop PSNR is
    measured on the object only — the (black) background is ~97% of the frame and would
    otherwise dominate the metric and hide convergence."""
    target_psnr: Optional[float] = None
    """If set, stop as soon as PSNR reaches this absolute value instead of a plateau."""
    camera_res_scale_factor: float = 0.25
    """Downscale factor for images/intrinsics (speed)."""
    snap_strays: bool = True
    """Apply the pipeline's per-frame hull snapping (reclaim strays onto the object) in BOTH
    conditions, so the experiment reflects the current method and isolates the effect of
    velocity prediction on top of it."""
    bg_subtract_mode: str = "threshold"
    """Segmentation mode passed to the dataparser ('threshold' or 'hysteresis')."""
    bg_subtract_threshold: float = 0.06
    bg_subtract_dir: str = "images_bgsub_t06e2"
    snap_distance_margin: float = 0.0
    """Distance-field snap margin (hull-voxel units) passed to the model."""
    temporal_smoothness_lambda: float = 0.0
    """Temporal attribute-smoothness weight passed to the model."""
    output_dir: Path = Path("experiments/outputs")
    """Where the histogram PNG and summary JSON are written."""
    device: str = "cuda"
    seed: int = 0


GAUSS = ("means", "scales", "quats", "opacities", "features_dc", "features_rest")
GEOM = ("means", "scales", "quats", "opacities")
COLOR = ("features_dc", "features_rest")


def _set_trainable(model, names, flag: bool) -> None:
    for n in names:
        model.gauss_params[n].requires_grad_(flag)


def _reset_adam(opt: Optimizers) -> None:
    for o in opt.optimizers.values():
        o.state.clear()


def _gt_for(model, image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    return model.composite_with_background(model.get_gt_img(image), background)


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR (dB) computed over the foreground pixels only (pred/gt in [0,1], [H,W,3])."""
    diff = (pred - gt)[mask]
    mse = (diff**2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def fit_frame(
    model,
    opt: Optimizers,
    cams: List,
    imgs: List[torch.Tensor],
    gts: Optional[List[torch.Tensor]],
    masks: Optional[List[torch.Tensor]],
    max_steps: int,
    stopper: Optional[EarlyStopper],
    base_step: int,
) -> Tuple[int, float]:
    """Optimize the currently-loaded frame using all views each step. Returns
    (steps_taken, final_object_psnr). If ``stopper`` is given, convergence is decided on the
    mean object-region PSNR and the loop stops as soon as it fires."""
    last_psnr = 0.0
    for s in range(1, max_steps + 1):
        model.step = base_step + s
        model.optimizers = opt.optimizers
        opt.zero_grad_all()
        loss = 0.0
        psnrs = []
        for i, (cam, img) in enumerate(zip(cams, imgs)):
            out = model.get_outputs(cam)
            loss = loss + sum(model.get_loss_dict(out, {"image": img}).values())
            if stopper is not None:
                with torch.no_grad():
                    psnrs.append(masked_psnr(out["rgb"], gts[i], masks[i]))
        loss.backward()
        opt.optimizer_step_all()
        if stopper is not None:
            last_psnr = float(np.mean(psnrs))
            if stopper.update(last_psnr, s):
                return s, last_psnr
    return max_steps, last_psnr


def track_sequence(model, dm, cfg: ExperimentConfig, use_velocity: bool) -> Tuple[List[int], List[float]]:
    """Track frames 1..N from the current (frame-0) state, returning per-frame
    (steps_to_convergence, converged_psnr)."""
    num_frames = min(cfg.num_frames, dm.num_frames)
    model._datamanager = dm  # so the model's hull-snapping can read per-frame cameras/masks
    opt = Optimizers(model_optimizer_config(), model.get_param_groups())
    steps: List[int] = []
    psnrs: List[float] = []
    prev = model.means.detach().clone()
    prev_prev: Optional[torch.Tensor] = None

    bg_black = torch.zeros(3, device=cfg.device)
    for frame in range(1, num_frames):
        views = dm.frame_to_indices[frame]
        imgs = [dm.cached_train[i]["image"].to(cfg.device) for i in views]
        cams = [dm.train_cameras[i : i + 1].to(cfg.device) for i in views]
        gts = [_gt_for(model, img, bg_black) for img in imgs]
        masks = [g[..., :3].mean(-1) > cfg.fg_threshold for g in gts]

        if use_velocity and prev_prev is not None and prev.shape == prev_prev.shape == model.means.shape:
            with torch.no_grad():
                model.means.data.add_(prev - prev_prev)

        # Pipeline default: snap strays back onto the visual hull (applied in both conditions).
        # Pointing the model's _prev_means at our `prev` lets its built-in history reset run, so
        # snapped gaussians don't get a spurious large velocity next frame.
        if cfg.snap_strays:
            with torch.no_grad():
                model._prev_means = prev
                model._snap_strays_to_hull(frame)

        _set_trainable(model, COLOR, False)  # freeze colours during tracking
        _set_trainable(model, GEOM, True)
        _reset_adam(opt)  # fresh per-frame optimizer momentum (matches train_video.py)

        stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_delta, target=cfg.target_psnr)
        n_steps, psnr = fit_frame(
            model, opt, cams, imgs, gts, masks, cfg.max_track_steps, stopper,
            base_step=model.config.initial_iterations,
        )
        steps.append(n_steps)
        psnrs.append(psnr)
        prev_prev = prev
        prev = model.means.detach().clone()
        CONSOLE.log(
            f"[exp] {'vel' if use_velocity else 'no-vel'} frame {frame}/{num_frames - 1}: "
            f"{n_steps} steps -> PSNR {psnr:.2f}"
        )
    return steps, psnrs


# method optimizer config is reused for every fresh Optimizers we build
_OPT_CFG = None


def model_optimizer_config():
    return _OPT_CFG.copy()


def main(cfg: ExperimentConfig) -> None:
    global _OPT_CFG
    torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = all_methods["temporal-splatfacto"]
    trainer_cfg.pipeline.datamanager.data = cfg.data
    trainer_cfg.pipeline.datamanager.dataparser.data = cfg.data
    trainer_cfg.pipeline.datamanager.dataparser.max_frames = cfg.num_frames
    trainer_cfg.pipeline.datamanager.camera_res_scale_factor = cfg.camera_res_scale_factor
    trainer_cfg.pipeline.datamanager.dataparser.bg_subtract_mode = cfg.bg_subtract_mode
    trainer_cfg.pipeline.datamanager.dataparser.bg_subtract_threshold = cfg.bg_subtract_threshold
    trainer_cfg.pipeline.datamanager.dataparser.bg_subtract_dir = cfg.bg_subtract_dir
    trainer_cfg.pipeline.model.initial_iterations = cfg.initial_iterations
    trainer_cfg.pipeline.model.tracking_iterations = 1
    trainer_cfg.pipeline.model.background_color = "black"
    trainer_cfg.pipeline.model.snap_distance_margin = cfg.snap_distance_margin
    trainer_cfg.pipeline.model.temporal_smoothness_lambda = cfg.temporal_smoothness_lambda
    _OPT_CFG = trainer_cfg.optimizers

    pipeline = trainer_cfg.pipeline.setup(device=cfg.device, test_mode="val", world_size=1, local_rank=0)
    model = pipeline.model
    dm = pipeline.datamanager

    # ---- fit frame 0 once (shared starting point), then snapshot ----
    views0 = dm.frame_to_indices[0]
    imgs0 = [dm.cached_train[i]["image"].to(cfg.device) for i in views0]
    cams0 = [dm.train_cameras[i : i + 1].to(cfg.device) for i in views0]
    model.train()
    _set_trainable(model, GAUSS, True)
    opt0 = Optimizers(model_optimizer_config(), model.get_param_groups())
    CONSOLE.log(f"[exp] fitting frame 0 for {cfg.initial_iterations} steps ({model.num_points} gaussians)...")
    fit_frame(model, opt0, cams0, imgs0, None, None, cfg.initial_iterations, stopper=None, base_step=0)
    snapshot = {n: p.detach().clone() for n, p in model.gauss_params.items()}

    def restore():
        with torch.no_grad():
            for n, p in model.gauss_params.items():
                p.data.copy_(snapshot[n])
        model._colors_frozen = False

    # ---- run both conditions from the identical frame-0 state ----
    results: Dict[str, Dict] = {}
    for use_velocity in (False, True):
        restore()
        steps, psnrs = track_sequence(model, dm, cfg, use_velocity=use_velocity)
        key = "velocity" if use_velocity else "no_velocity"
        censored = int(np.sum(np.asarray(steps) >= cfg.max_track_steps))
        results[key] = {
            "steps": steps,
            "psnr": psnrs,
            "mean_steps": float(np.mean(steps)),
            "censored": censored,
        }
        CONSOLE.print(
            f"[bold]{key}[/bold]: mean steps={np.mean(steps):.1f}  median={np.median(steps):.0f}  "
            f"mean converged PSNR={np.mean(psnrs):.2f}"
        )

    plot_path = cfg.output_dir / "velocity_convergence_histogram.png"
    plot_histogram(results, plot_path, cfg)
    summary = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "results": results,
        "mean_step_reduction_pct": 100.0
        * (results["no_velocity"]["mean_steps"] - results["velocity"]["mean_steps"])
        / max(1e-9, results["no_velocity"]["mean_steps"]),
    }
    (cfg.output_dir / "velocity_convergence_summary.json").write_text(json.dumps(summary, indent=2))
    CONSOLE.print(
        f"\n[green]Saved[/green] {plot_path}\nMean steps: "
        f"no-velocity={results['no_velocity']['mean_steps']:.1f}, "
        f"velocity={results['velocity']['mean_steps']:.1f} "
        f"({summary['mean_step_reduction_pct']:.1f}% fewer with velocity)."
    )


def plot_histogram(results: Dict[str, Dict], path: Path, cfg: ExperimentConfig) -> None:
    nov = np.asarray(results["no_velocity"]["steps"])
    vel = np.asarray(results["velocity"]["steps"])
    # Fixed bin count over the shared range so both distributions (velocity is far tighter)
    # are visible rather than velocity collapsing into a single bar.
    bins = np.linspace(0, float(max(nov.max(), vel.max())) * 1.02, 31)

    c_nov, c_vel = "#d1495b", "#1b6ca8"  # warm red vs cool blue
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for data, color, label in ((nov, c_nov, "without velocity"), (vel, c_vel, "with velocity")):
        ax.hist(data, bins=bins, color=color, alpha=0.55, edgecolor="white", linewidth=0.8, label=label, zorder=2)

    for data, color, label in ((nov, c_nov, "without velocity"), (vel, c_vel, "with velocity")):
        m = float(np.mean(data))
        ax.axvline(m, color=color, linestyle="--", linewidth=2.2, zorder=3)
        ax.text(
            m, ax.get_ylim()[1] * (0.96 if label == "without velocity" else 0.86),
            f"  mean = {m:.0f}", color=color, fontweight="bold", va="top", ha="left",
        )

    reduction = 100.0 * (nov.mean() - vel.mean()) / max(1e-9, nov.mean())
    if cfg.target_psnr is not None:
        criterion = f"early stop: object PSNR ≥ {cfg.target_psnr:.1f} dB"
        metric_axis = f"Steps to reach {cfg.target_psnr:.1f} dB object PSNR"
    else:
        criterion = f"early stop: PSNR plateau, patience={cfg.patience}, Δ<{cfg.min_delta} dB"
        metric_axis = "Steps to convergence (PSNR plateau)"
    cens_nov = int(np.sum(nov >= cfg.max_track_steps))
    cens_vel = int(np.sum(vel >= cfg.max_track_steps))
    cens_note = ""
    if cens_nov or cens_vel:
        cens_note = f"; not reached in {cfg.max_track_steps} steps: no-vel {cens_nov}, vel {cens_vel} (shown at cap)"
    ax.set_title(
        "Per-frame steps to convergence — velocity prediction vs none\n"
        f"({criterion}; {len(nov)} frames; "
        f"velocity needs {reduction:.0f}% fewer steps on average{cens_note})",
        fontsize=11.5,
    )
    ax.set_xlabel(metric_axis)
    ax.set_ylabel("Number of frames")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(ExperimentConfig))
