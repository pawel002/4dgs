# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Temporal (4D) gaussian splatting on top of splatfacto.

This is a nerfstudio port of the ``train_video.py`` algorithm from the 4DGS
thesis repo. The scene is observed by a small set of *fixed* cameras while it
deforms over time; we reconstruct one gaussian cloud per time step.

Algorithm
---------
* **Frame 0 (initial reconstruction).** Run normal splatfacto with densification
  for ``initial_iterations`` steps. This both fits the geometry and *fixes the
  gaussian count* for the rest of the video.
* **Frames 1..T (tracking).** For each subsequent frame, refine for
  ``tracking_iterations`` steps with densification disabled (the gaussian count
  never changes again). Two optimisations carry over from ``train_video.py``:

    1. **Linear velocity prediction** — at each frame transition the gaussians are
       extrapolated ``x_t += (x_{t-1} - x_{t-2})``, giving the optimiser a much
       better starting point when the motion is smooth.
    2. **Colour freezing** — the SH colour coefficients learned on frame 0 are
       frozen, so tracking only moves/reshapes gaussians instead of re-fitting
       appearance (faster, and avoids colour drift).

  The means learning rate is also held at a constant ``tracking_means_lr`` during
  tracking (rather than the long global decay used for a single static scene),
  so late frames can still move.

Everything else (rasterisation, losses, the viewer, nerfstudio ``.ckpt``
checkpoints, ``ns-export gaussian-splat``) is inherited unchanged from
:class:`SplatfactoModel`. In addition, the converged gaussians of every frame are
written to ``<output>/temporal_frames/<frame>.ply`` as the primary temporal
artifact (see :mod:`nerfstudio.engine.temporal_trainer`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy

from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.temporal_schedule import EarlyStopper, TemporalSchedule
from nerfstudio.viewer.viewer_elements import ViewerControl


@dataclass
class TemporalSplatfactoModelConfig(SplatfactoModelConfig):
    """Temporal Splatfacto config. Inherits every splatfacto field and adds the
    temporal schedule / tracking behaviour."""

    _target: Type = field(default_factory=lambda: TemporalSplatfactoModel)
    initial_iterations: int = 3000
    """Steps spent on the dense reconstruction of frame 0 (with densification)."""
    tracking_iterations: int = 300
    """Steps spent tracking each subsequent frame (no densification)."""
    use_velocity_prediction: bool = True
    """Extrapolate gaussian positions across frames using linear velocity."""
    velocity_mask_gating: bool = True
    """Only apply velocity extrapolation to gaussians that still project ONTO the object
    silhouette; zero the velocity of strays. Fixes 'runaway velocity': a gaussian that drifts
    off the object fades and stops receiving a corrective gradient, so its frame-to-frame
    displacement (and hence its extrapolated velocity) freezes at a constant value and it
    coasts away forever. Gating detects such off-object gaussians via the masks and stops
    extrapolating them, so they no longer fly off (the silhouette loss then fades them out)."""
    velocity_gate_min_view_fraction: float = 0.5
    """A gaussian keeps its velocity only if its center projects inside the object silhouette
    in at least this fraction of the cameras. A true runaway projects outside in (almost) every
    view, so a majority vote (0.5) flags it; using a *fraction* rather than 'inside in every
    camera' avoids freezing legitimate gaussians that fall in a below-threshold (dark) hole of
    the mask in one or two views. (Only used when ``snap_strays_to_hull`` is off.)"""
    snap_strays_to_hull: bool = True
    """At the start of every frame, after applying velocity, snap any gaussian that has ended
    up OUTSIDE the mask-intersection (visual hull) back onto its nearest hull-surface voxel.
    This *reclaims* drifted gaussians instead of leaving them behind: every frame begins with
    all gaussians inside the object as seen from all cameras, so they must align to it. Replaces
    the milder 'freeze stray velocity' gating (which only stops strays without recovering them)."""
    snap_min_view_fraction: float = 1.0
    """A gaussian is snapped if it projects inside the object in FEWER than this fraction of
    cameras. Default 1.0 = snap everything not inside *every* mask (the strict mask-intersection
    / 'inside all camera object maps'); this re-registers the whole cloud onto the current
    frame's object each frame and best maintains coverage. Lower it (e.g. 0.5) to snap only
    gaussians that are outside in a majority of views, which is gentler but leaves more behind.
    Note: with the strict default, gaussians sitting in a dark, below-threshold region of the
    object (which the brightness mask excludes, and which carries no photometric signal anyway)
    are also pushed to the lit surface."""
    hull_snap_resolution: int = 96
    """Voxel-grid resolution per axis for carving the per-frame hull used as snap targets."""
    hull_snap_max_points: int = 30000
    """Cap on hull voxels kept for nearest-surface snapping (subsampled if exceeded)."""
    snap_distance_margin: float = 0.0
    """Distance-field clamping: when > 0, a gaussian flagged by the projective stray test is
    snapped only if its 3D distance to the nearest hull voxel exceeds this many *hull voxel
    sizes*. The binary projective test treats a gaussian one pixel outside a noisy mask the
    same as one that coasted ten radii away; the margin distinguishes them — borderline
    gaussians (within the margin shell of the hull, typically flagged only because of residual
    silhouette holes) are left for the optimizer, while genuine runaways beyond the shell are
    still teleported back. 0 keeps the legacy behaviour (every flagged gaussian is snapped)."""
    freeze_colors_when_tracking: bool = True
    """Freeze SH colour coefficients (features_dc/features_rest) after frame 0. Keeps the
    per-gaussian colour constant across the video so tracking aligns gaussians to motion
    instead of re-fitting appearance (and so a gaussian cannot recolour to black to hide in
    the background)."""
    freeze_opacity_when_tracking: bool = False
    """Also freeze per-gaussian opacity after frame 0 (in addition to colour). With opacity
    fixed, a gaussian can no longer fade to ~0 to 'hide' in the background — it must MOVE with
    the object, which more strictly enforces the constant-gaussians/motion-only philosophy and
    further suppresses background fitting. Off by default to match the original train_video.py
    (which lets opacity adapt to occlusion/appearance changes between frames)."""
    temporal_smoothness_lambda: float = 0.0
    """Weight of the temporal attribute-smoothness regularizer. When > 0, every tracking step
    adds an L2 penalty pulling the *non-positional* per-gaussian attributes (opacity logits,
    log-scales, quaternions) towards their converged values of the previous frame. Positions
    are exempt (their inter-frame change is the genuine motion, already handled by the
    velocity warm start); the non-positional attributes of a rigid subject should evolve
    slowly, and their per-frame fluctuation under a generous tracking budget is optimizer
    noise, not signal. This is the temporal-compressor's smoothness prior moved into the
    optimizer: it damps frame-to-frame attribute flicker at the source, which both stabilizes
    renders and makes the attribute time series compressible. 0 disables (legacy behaviour)."""
    tracking_means_lr: float = 1.6e-4
    """Constant means learning rate used during tracking frames."""
    means_lr_init: float = 1.6e-4
    """Means learning rate at the start of the frame-0 reconstruction."""
    means_lr_final: float = 1.6e-6
    """Means learning rate at the end of the frame-0 reconstruction."""
    save_ply_per_frame: bool = True
    """Write each frame's converged gaussians to ``<output>/temporal_frames/<frame>.ply``."""

    # ---- foreground masking & silhouette (alpha) supervision ----
    # These keep the gaussians on the object and stop them drifting into / fitting the black
    # background over the video. The mask is derived from the GT image brightness because the
    # renders are segmented onto black but have an *opaque* alpha channel (so alpha carries no
    # silhouette — unlike the original train_video.py, whose data put the mask in alpha).
    use_foreground_mask: bool = True
    """Restrict the photometric (L1 + SSIM) loss to the object foreground, like the original
    train_video.py masked the rendered image by its alpha. Prevents the (large) black
    background from dominating / biasing the loss."""
    alpha_mask_loss_lambda: float = 1.0
    """Weight of the silhouette (alpha) supervision loss. This is the actual anti-drift term:
    it penalizes rendered opacity (accumulation) that falls OUTSIDE the object mask, so a
    gaussian cannot sit in the background without being pushed out / faded. Set 0 to disable."""
    alpha_inside_loss_lambda: float = 0.1
    """Weight pulling rendered opacity towards 1 INSIDE the mask (so the object stays solidly
    covered). Kept small so it nudges rather than forces. Set 0 to only penalize outside."""
    mask_threshold: float = 0.05
    """GT brightness (0-1, mean over RGB) above which a pixel is object foreground."""
    mask_dilate: int = 3
    """Dilate the foreground mask by this many pixels before use, so gaussians right at the
    silhouette edge are not over-penalized."""

    # ---- early stopping of per-frame tracking ----
    early_stop_enabled: bool = False
    """Stop refining a tracking frame once its reconstruction has converged (PSNR plateau),
    instead of always spending the full ``tracking_iterations``. Off by default so the fixed
    schedule is unchanged; the velocity experiment turns it on."""
    early_stop_patience: int = 30
    """Number of consecutive steps without a PSNR improvement (> ``early_stop_min_delta``)
    before a tracking frame is declared converged."""
    early_stop_min_delta: float = 0.03
    """Minimum PSNR (dB) increase that counts as an improvement for early stopping."""
    early_stop_target_psnr: Optional[float] = None
    """If set, a tracking frame stops as soon as its PSNR reaches this absolute value
    (instead of waiting for a plateau)."""
    early_stop_fg_threshold: float = 0.05
    """The early-stop PSNR is measured on the object foreground only (GT brightness above this
    value). The segmented background is ~most of the frame, so full-image PSNR would be
    dominated by it and never plateau; masking makes convergence on the object observable."""
    # Temporal sequences span many frames; the per-frame resolution schedule of
    # vanilla splatfacto would make early/late frames train at different
    # resolutions, so default to full resolution throughout.
    num_downscales: int = 0
    # The input images are segmented onto a black background, so render against
    # black too (instead of splatfacto's random background).
    background_color: Literal["random", "black", "white"] = "black"


class TemporalSplatfactoModel(SplatfactoModel):
    """Splatfacto with a per-frame temporal training schedule. See the module docstring.

    The initial gaussians are seeded from the object's *visual hull* — carved from the
    frame-0 camera silhouettes by :class:`TemporalDataParser` and passed in as
    ``seed_points`` — so the cloud starts tightly inside the object being reconstructed."""

    config: TemporalSplatfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

        # Number of frames is provided by the temporal dataparser via metadata; the
        # temporal trainer also sets it directly. Default to 1 (single frame) if absent.
        meta = self.kwargs.get("metadata", {}) or {}
        self.num_frames: int = int(meta.get("num_frames", 1))
        self.schedule = TemporalSchedule(
            num_frames=self.num_frames,
            initial_iterations=self.config.initial_iterations,
            tracking_iterations=self.config.tracking_iterations,
        )

        # Stop all densification/culling once frame 0 is done so the gaussian count
        # is fixed for the rest of the video.
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.refine_stop_iter = self.config.initial_iterations
            self.strategy.refine_scale2d_stop_iter = self.config.initial_iterations

        # Velocity-prediction state: converged means of the two previous frames.
        self._prev_means: Optional[torch.Tensor] = None
        self._prev_prev_means: Optional[torch.Tensor] = None
        self._colors_frozen: bool = False

        # Temporal-smoothness anchor: the previous frame's converged non-positional
        # attributes (opacity logits, log-scales, quats), towards which the regularizer pulls.
        self._smooth_anchor: Optional[dict] = None

        # Per-frame diagnostics (snap counts, velocity magnitude), dumped to
        # temporal_stats.json next to the per-frame plys.
        self.temporal_stats: dict = {"frames": {}}

        # Early-stopping state for per-frame tracking.
        self.early_stopper = EarlyStopper(
            patience=self.config.early_stop_patience,
            min_delta=self.config.early_stop_min_delta,
            target=self.config.early_stop_target_psnr,
        )
        self._frame_converged: bool = False
        self._last_psnr: Optional[float] = None
        self.convergence_steps: dict = {}  # frame index -> local step at convergence

        # Where per-frame plys are written; set by the temporal trainer.
        self.temporal_ply_dir: Optional[Path] = None

        # Live "which frame is being aligned" readout in the viser viewer. We use a
        # ViewerControl to reach the raw viser server and create our own gui handle:
        # nerfstudio wraps every managed ViewerElement's update callback in the train
        # lock, so setting a managed element's value from inside the training loop
        # (which already holds that lock) would deadlock. A handle we create directly
        # has no such callback, so updating it every step is safe.
        self.viewer_control = ViewerControl()
        self._frame_handle = None

    # ------------------------------------------------------------------ helpers
    def set_schedule(self, schedule: TemporalSchedule) -> None:
        """Adopt the (authoritative) schedule built by the temporal trainer."""
        self.schedule = schedule
        self.num_frames = schedule.num_frames
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.refine_stop_iter = schedule.initial_iterations
            self.strategy.refine_scale2d_stop_iter = schedule.initial_iterations

    def _frame_status_text(self, frame: int, step: int) -> str:
        phase = "reconstructing" if frame == 0 else "tracking"
        return f"frame {frame + 1}/{self.num_frames} · {phase} (step {step})"

    def _update_frame_readout(self, frame: int, step: int) -> None:
        """Show the current frame in the viser viewer (no-op if the viewer is off)."""
        server = getattr(self.viewer_control, "viser_server", None)
        if server is None:
            return
        text = self._frame_status_text(frame, step)
        try:
            if self._frame_handle is None:
                self._frame_handle = server.gui.add_text("Temporal frame", initial_value=text, disabled=True)
            else:
                self._frame_handle.value = text
        except Exception:  # pragma: no cover - viewer is best-effort, never break training
            pass

    def _set_means_lr(self, step: int) -> None:
        """Phase-aware learning rate for the gaussian means.

        Frame 0 decays exponentially (like a normal splat warmup); tracking frames
        use a constant rate so even late frames retain enough step size to move.
        """
        if not hasattr(self, "optimizers") or "means" not in self.optimizers:
            return
        if self.config.early_stop_enabled and self._frame_converged:
            lr = 0.0  # frame has early-stopped; keep it frozen for the rest of its budget
        elif self.schedule.is_initial_frame(step):
            frac = min(1.0, step / max(1, self.config.initial_iterations))
            lr = self.config.means_lr_init * (self.config.means_lr_final / self.config.means_lr_init) ** frac
        else:
            lr = self.config.tracking_means_lr
        for group in self.optimizers["means"].param_groups:
            group["lr"] = lr

    def _freeze_appearance(self) -> None:
        """Freeze the per-gaussian appearance params for tracking: always colour, and
        optionally opacity. Keeps these constant across frames so tracking aligns gaussians to
        motion (and removes the 'fade to black background' escape hatch when opacity is frozen)."""
        if self._colors_frozen:
            return
        frozen = []
        if self.config.freeze_colors_when_tracking:
            frozen += ["features_dc", "features_rest"]
        if self.config.freeze_opacity_when_tracking:
            frozen += ["opacities"]
        for name in frozen:
            self.gauss_params[name].requires_grad_(False)
        self._colors_frozen = True
        if frozen:
            CONSOLE.log(f"[temporal] froze for tracking: {', '.join(frozen)}.")

    def _freeze_geometry_via_lr(self) -> None:
        """Stop a converged tracking frame from moving by zeroing the gaussian learning rates.

        We zero LRs rather than flipping ``requires_grad`` because gsplat's strategy calls
        ``retain_grad()`` on the projected means every forward pass, which errors if the means
        no longer require grad. With LR=0 the optimizer simply makes no update. ``means`` is
        kept at 0 by ``_set_means_lr`` while ``_frame_converged`` is set."""
        if not hasattr(self, "optimizers"):
            return
        for name in ("scales", "quats", "opacities"):
            if name in self.optimizers:
                for group in self.optimizers[name].param_groups:
                    group["lr"] = 0.0

    def _unfreeze_geometry(self) -> None:
        """Restore non-zero geometry learning rates at the start of a new tracking frame
        (``means`` is re-set per-step by ``_set_means_lr``; colours stay frozen by design)."""
        if not hasattr(self, "optimizers"):
            return
        for name, lr in (("scales", 0.005), ("quats", 0.001), ("opacities", 0.05)):
            if name in self.optimizers:
                for group in self.optimizers[name].param_groups:
                    group["lr"] = lr

    def get_metrics_dict(self, outputs, batch):  # type: ignore[override]
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Stash an object-region PSNR so the early stopper (run in the next step's callback)
        # can read it. Masked rather than full-image: the black background is most of the
        # frame and would otherwise dominate PSNR and hide convergence on the object.
        if self.training and self.config.early_stop_enabled:
            with torch.no_grad():
                gt = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
                pred = outputs["rgb"]
                fg = gt[..., :3].mean(-1) > self.config.early_stop_fg_threshold
                if fg.any():
                    mse = ((pred - gt)[fg] ** 2).mean().clamp_min(1e-12)
                    self._last_psnr = float(-10.0 * torch.log10(mse))
                elif "psnr" in metrics_dict:
                    self._last_psnr = float(metrics_dict["psnr"])
        return metrics_dict

    # ------------------------------------------------------------------ foreground mask
    def _dilate(self, mask: torch.Tensor, d: int) -> torch.Tensor:
        if d <= 0:
            return mask
        pooled = F.max_pool2d(mask.permute(2, 0, 1)[None], kernel_size=2 * d + 1, stride=1, padding=d)
        return pooled[0].permute(1, 2, 0)

    def _erode(self, mask: torch.Tensor, d: int) -> torch.Tensor:
        if d <= 0:
            return mask
        return 1.0 - self._dilate(1.0 - mask, d)

    def foreground_masks(self, gt_img: torch.Tensor):
        """Three silhouette regions from GT brightness ([H, W, 1] floats in {0,1}):

        * ``core``    — eroded object interior; the α→1 (inside) supervision target.
        * ``fg``      — the object, dilated; used to include the whole object in the
          photometric loss.
        * ``outside`` — everything beyond the dilated object; the α→0 (anti-drift) region.

        The band between the raw silhouette and its dilation is a "don't care" zone so gaussian
        tails right at the edge are neither forced opaque nor penalized. (Brightness, not alpha:
        the renders are segmented onto black but have an opaque alpha channel.)
        """
        raw = (gt_img[..., :3].mean(dim=-1, keepdim=True) > self.config.mask_threshold).float()
        d = self.config.mask_dilate
        fg = self._dilate(raw, d)
        core = self._erode(raw, d)
        outside = 1.0 - fg
        return core, fg, outside

    def get_loss_dict(self, outputs, batch, metrics_dict=None):  # type: ignore[override]
        """Photometric loss restricted to the object + silhouette (alpha) supervision.

        Two additions over splatfacto keep the gaussians on the object and stop them drifting
        into the black background over the video:

        * **Foreground-masked photometric loss** — the L1 + SSIM terms are computed on the
          object region only (rendered & GT images multiplied by the object mask), mirroring
          the original train_video.py's ``image * alpha_mask``. Because our renders have an
          opaque alpha, the mask is derived from GT brightness instead of the alpha channel.
        * **Silhouette (alpha) supervision** — the rendered accumulation (opacity) is pushed
          to 0 *clearly outside* the object and (gently) to 1 in the object *core*. This is the
          real anti-drift force: a gaussian that wanders into the background renders nonzero
          opacity there and is immediately penalized, so it is pushed back onto the object or
          faded out. Without it, a dark gaussian in the background matches the black background
          at ~0 photometric cost and is free to drift.
        """
        if not self.config.use_foreground_mask and self.config.alpha_mask_loss_lambda == 0.0:
            return super().get_loss_dict(outputs, batch, metrics_dict)

        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        core, fg, outside = self.foreground_masks(gt_img)

        # Photometric loss on the (dilated) object region, via the base implementation.
        if self.config.use_foreground_mask:
            batch = {**batch, "mask": fg}
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Silhouette / alpha supervision (the band between `core` and `fg` is don't-care).
        if self.config.alpha_mask_loss_lambda > 0.0 or self.config.alpha_inside_loss_lambda > 0.0:
            accum = outputs["accumulation"]  # [H, W, 1], rendered opacity in [0, 1]
            if self.config.alpha_mask_loss_lambda > 0.0:
                loss_dict["alpha_outside"] = (
                    self.config.alpha_mask_loss_lambda * (accum * outside).sum() / outside.sum().clamp_min(1.0)
                )
            if self.config.alpha_inside_loss_lambda > 0.0:
                loss_dict["alpha_inside"] = (
                    self.config.alpha_inside_loss_lambda * ((1.0 - accum) * core).sum() / core.sum().clamp_min(1.0)
                )

        # Temporal attribute smoothness: pull the non-positional attributes towards their
        # previous-frame converged values (the compressor's smoothness prior, applied at the
        # source). Active only while tracking and only when an anchor of matching shape exists.
        if self.training and self.config.temporal_smoothness_lambda > 0.0 and self._smooth_anchor is not None:
            lam = self.config.temporal_smoothness_lambda
            smooth = None
            for name in ("opacities", "scales", "quats"):
                anchor = self._smooth_anchor.get(name)
                param = self.gauss_params[name]
                if anchor is None or anchor.shape != param.shape:
                    continue
                term = ((param - anchor) ** 2).mean()
                smooth = term if smooth is None else smooth + term
            if smooth is not None:
                loss_dict["temporal_smoothness"] = lam * smooth
        return loss_dict

    # ------------------------------------------------------------------ on-object test
    def _frame_stat(self, frame_idx: int, key: str, value) -> None:
        """Record a per-frame diagnostic (written to temporal_stats.json at frame save)."""
        self.temporal_stats["frames"].setdefault(str(frame_idx), {})[key] = value

    @torch.no_grad()
    def _frame_view_masks(self, frame_idx: int):
        """List of (viewmat, K, mask2d, H, W) for every camera of ``frame_idx``; the dilated
        object foreground mask per view. Returns [] if the datamanager is unavailable."""
        dm = getattr(self, "_datamanager", None)
        if dm is None or not hasattr(dm, "frame_to_indices"):
            return []
        views = dm.frame_to_indices.get(frame_idx, [])
        out = []
        device = self.means.device
        black = torch.zeros(3, device=device)
        for vi in views:
            cam = dm.train_cameras[vi : vi + 1].to(device)
            img = dm.cached_train[vi]["image"].to(device)
            gt = self.composite_with_background(self.get_gt_img(img), black)
            _, fg, _ = self.foreground_masks(gt)  # dilated object mask [H, W, 1]
            viewmat = get_viewmat(cam.camera_to_worlds.reshape(1, 3, 4))[0]
            K = cam.get_intrinsics_matrices().reshape(3, 3).to(device)
            out.append((viewmat, K, fg[..., 0], fg.shape[0], fg.shape[1]))
        return out

    @torch.no_grad()
    def _count_inside(self, view_masks, pts: torch.Tensor) -> torch.Tensor:
        """For each point, count how many of the cameras' object masks it projects inside."""
        n = pts.shape[0]
        votes = torch.zeros(n, dtype=torch.int32, device=pts.device)
        for viewmat, K, mask, H, W in view_masks:
            cp = pts @ viewmat[:3, :3].T + viewmat[:3, 3]
            z = cp[:, 2]
            proj = cp @ K.T
            u = (proj[:, 0] / z).round().long()
            v = (proj[:, 1] / z).round().long()
            in_view = (z > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            sel = in_view.nonzero(as_tuple=True)[0]
            votes[sel] += (mask[v[sel], u[sel]] > 0.5).int()
        return votes

    @torch.no_grad()
    def _on_object(self, frame_idx: int):
        """Boolean [N]: gaussians whose center projects inside the object silhouette in at least
        ``velocity_gate_min_view_fraction`` of ``frame_idx``'s cameras (still on the object).
        Returns None if the per-frame views are unavailable (fall back to ungated velocity)."""
        vm = self._frame_view_masks(frame_idx)
        if not vm:
            return None
        needed = max(1, int(round(self.config.velocity_gate_min_view_fraction * len(vm))))
        return self._count_inside(vm, self.means.detach()) >= needed

    @torch.no_grad()
    def _carve_hull(self, view_masks, means: torch.Tensor):
        """Carve the visual hull (points inside EVERY camera mask) on a voxel grid spanning the
        current gaussian cloud. Returns ``(hull_points, voxel_size)`` with up to
        ``hull_snap_max_points`` voxel centers; ``voxel_size`` is the grid spacing (the length
        unit of ``snap_distance_margin``)."""
        lo = torch.quantile(means, 0.01, dim=0)
        hi = torch.quantile(means, 0.99, dim=0)
        center = (lo + hi) / 2.0
        half = float((hi - lo).max()) / 2.0 * 1.2
        res = self.config.hull_snap_resolution
        lin = torch.linspace(-half, half, res, device=means.device)
        voxel_size = float(2.0 * half / max(res - 1, 1))
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1) + center
        votes = self._count_inside(view_masks, grid)
        hull = grid[votes >= len(view_masks)]
        cap = self.config.hull_snap_max_points
        if hull.shape[0] > cap:
            hull = hull[torch.randperm(hull.shape[0], device=hull.device)[:cap]]
        return hull, voxel_size

    @torch.no_grad()
    def _snap_strays_to_hull(self, frame_idx: int) -> int:
        """Snap every gaussian that is OUTSIDE the mask-intersection (visual hull) of
        ``frame_idx`` onto its nearest hull-surface voxel, so the frame starts with all
        gaussians inside the object. Also resets the velocity history of snapped gaussians so
        the teleport is not counted as motion next frame. Returns the number snapped.

        With ``snap_distance_margin > 0`` the projective stray test only *nominates*
        candidates; a candidate is snapped only if it lies farther from the hull (in 3D) than
        the margin, so borderline gaussians flagged by residual mask holes are left alone."""
        vm = self._frame_view_masks(frame_idx)
        if not vm:
            return 0
        means = self.means.detach()
        # A gaussian is a stray if it is inside fewer than `snap_min_view_fraction` of the
        # masks (default: not inside ALL of them — the strict mask-intersection).
        votes = self._count_inside(vm, means)
        needed = max(1, int(round(self.config.snap_min_view_fraction * len(vm))))
        stray_idx = (votes < needed).nonzero(as_tuple=True)[0]
        n_flagged = int(stray_idx.numel())
        self._frame_stat(frame_idx, "flagged", n_flagged)
        if stray_idx.numel() == 0:
            return 0
        hull, voxel_size = self._carve_hull(vm, means)
        if hull.shape[0] == 0:
            return 0
        # nearest hull voxel per stray (chunked to bound memory)
        snapped = torch.empty((stray_idx.numel(), 3), device=means.device)
        nn_dist = torch.empty(stray_idx.numel(), device=means.device)
        pts = means[stray_idx]
        for s in range(0, pts.shape[0], 4096):
            chunk = pts[s : s + 4096]
            d = torch.cdist(chunk, hull)
            mins = d.min(dim=1)
            snapped[s : s + 4096] = hull[mins.indices]
            nn_dist[s : s + 4096] = mins.values
        if self.config.snap_distance_margin > 0.0:
            far = nn_dist > self.config.snap_distance_margin * voxel_size
            stray_idx, snapped = stray_idx[far], snapped[far]
            if stray_idx.numel() == 0:
                self._frame_stat(frame_idx, "snapped", 0)
                return 0
        self.means.data[stray_idx] = snapped
        if self._prev_means is not None and self._prev_means.shape == self.means.shape:
            self._prev_means[stray_idx] = snapped  # so next-frame velocity from here is ~0
        self._frame_stat(frame_idx, "snapped", int(stray_idx.numel()))
        return int(stray_idx.numel())

    # ------------------------------------------------------------------ callbacks
    def get_training_callbacks(self, training_callback_attributes):  # type: ignore[override]
        # Capture the datamanager so the velocity callback can read per-frame cameras/masks.
        pipeline = getattr(training_callback_attributes, "pipeline", None)
        if pipeline is not None:
            self._datamanager = getattr(pipeline, "datamanager", None)
        return super().get_training_callbacks(training_callback_attributes)

    def step_cb(self, optimizers: Optimizers, step):  # type: ignore[override]
        # Base class records self.step / self.optimizers / self.schedulers.
        super().step_cb(optimizers, step)
        frame = self.schedule.frame_of_step(step)

        # On a frame transition into a tracking frame, apply the temporal optimisations.
        if self.schedule.is_first_step_of_frame(step) and frame >= 1:
            if self.config.freeze_colors_when_tracking or self.config.freeze_opacity_when_tracking:
                self._freeze_appearance()
            if self.config.early_stop_enabled:
                # New frame: re-arm early stopping and un-freeze geometry from the previous frame.
                self.early_stopper.reset()
                self._frame_converged = False
                self._last_psnr = None
                self._unfreeze_geometry()
            if (
                self.config.use_velocity_prediction
                and self._prev_means is not None
                and self._prev_prev_means is not None
                and self._prev_means.shape == self._prev_prev_means.shape == self.means.shape
            ):
                with torch.no_grad():
                    velocity = self._prev_means - self._prev_prev_means
                    gated_note = ""
                    # Gating freezes stray velocity; only used when we are NOT snapping strays
                    # back (snapping reclaims them, which is strictly stronger).
                    if self.config.velocity_mask_gating and not self.config.snap_strays_to_hull:
                        on_object = self._on_object(frame - 1)
                        if on_object is not None:
                            velocity = velocity * on_object[:, None].to(velocity.dtype)
                            gated_note = f", froze velocity for {int((~on_object).sum())}/{on_object.numel()} strays"
                    self.means.data.add_(velocity)
                    self._frame_stat(frame, "velocity_mean", float(velocity.norm(dim=1).mean()))
                CONSOLE.log(
                    f"[temporal] frame {frame + 1}/{self.num_frames}: velocity prediction "
                    f"(mean |Δx|={velocity.norm(dim=1).mean().item():.6f}{gated_note})."
                )

            # Snap any gaussian that ended up outside the visual hull back onto the object, so
            # the frame starts with every gaussian inside all camera masks (reclaims strays).
            if self.config.snap_strays_to_hull:
                with torch.no_grad():
                    n_snapped = self._snap_strays_to_hull(frame)
                if n_snapped:
                    CONSOLE.log(
                        f"[temporal] frame {frame + 1}/{self.num_frames}: snapped {n_snapped}/"
                        f"{self.num_points} strays onto the visual hull."
                    )

        # Early stopping: once a tracking frame's PSNR has converged, freeze its gaussians for
        # the remaining budgeted steps (the next frame un-freezes them).
        if (
            self.config.early_stop_enabled
            and frame >= 1
            and not self._frame_converged
            and self._last_psnr is not None
            and not self.schedule.is_first_step_of_frame(step)
        ):
            if self.early_stopper.update(self._last_psnr, self.schedule.frame_local_step(step)):
                self._frame_converged = True
                self.convergence_steps[frame] = self.schedule.frame_local_step(step)
                self._freeze_geometry_via_lr()
                CONSOLE.log(
                    f"[temporal] frame {frame + 1}/{self.num_frames}: early-stopped at "
                    f"{self.convergence_steps[frame]} steps (PSNR={self._last_psnr:.2f})."
                )

        self._set_means_lr(step)

        # Update the live viewer readout.
        self._update_frame_readout(frame, step)

    def step_post_backward(self, step):  # type: ignore[override]
        super().step_post_backward(step)
        if not self.schedule.is_last_step_of_frame(step):
            return
        frame = self.schedule.frame_of_step(step)
        # Snapshot converged means for velocity prediction of the next frames, and the
        # converged non-positional attributes as the next frame's smoothness anchor.
        with torch.no_grad():
            self._prev_prev_means = self._prev_means
            self._prev_means = self.means.detach().clone()
            if self.config.temporal_smoothness_lambda > 0.0:
                self._smooth_anchor = {
                    name: self.gauss_params[name].detach().clone() for name in ("opacities", "scales", "quats")
                }
        if self.config.save_ply_per_frame:
            self._save_frame_ply(frame)
        self._save_temporal_stats()

    def _save_temporal_stats(self) -> None:
        """Dump the per-frame diagnostics (flagged/snapped counts, ...) next to the plys."""
        if self.temporal_ply_dir is None or not self.temporal_stats["frames"]:
            return
        import json

        out_dir = Path(self.temporal_ply_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            (out_dir / "temporal_stats.json").write_text(json.dumps(self.temporal_stats, indent=1))
        except OSError:  # diagnostics are best-effort, never break training
            pass

    # ------------------------------------------------------------------ per-frame ply
    @torch.no_grad()
    def _gaussian_ply_tensors(self) -> "OrderedDict[str, np.ndarray]":
        """Build the same per-gaussian property map that ``ns-export gaussian-splat``
        writes (sh_coeffs mode), so per-frame plys load in any nerfstudio/3DGS viewer."""
        m = OrderedDict()
        positions = self.means.detach().cpu().numpy()
        n = positions.shape[0]
        m["x"], m["y"], m["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
        m["nx"] = np.zeros(n, dtype=np.float32)
        m["ny"] = np.zeros(n, dtype=np.float32)
        m["nz"] = np.zeros(n, dtype=np.float32)

        shs_0 = self.shs_0.detach().contiguous().cpu().numpy()
        for i in range(shs_0.shape[1]):
            m[f"f_dc_{i}"] = shs_0[:, i, None]
        if self.config.sh_degree > 0:
            shs_rest = self.shs_rest.detach().transpose(1, 2).contiguous().cpu().numpy().reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                m[f"f_rest_{i}"] = shs_rest[:, i, None]

        m["opacity"] = self.opacities.detach().cpu().numpy()
        scales = self.scales.detach().cpu().numpy()
        for i in range(3):
            m[f"scale_{i}"] = scales[:, i, None]
        quats = self.quats.detach().cpu().numpy()
        for i in range(4):
            m[f"rot_{i}"] = quats[:, i, None]
        return m

    @torch.no_grad()
    def _save_frame_ply(self, frame: int) -> None:
        if self.temporal_ply_dir is None:
            return
        # Imported lazily: the exporter module pulls in heavy optional deps.
        from nerfstudio.scripts.exporter import ExportGaussianSplat

        out_dir = Path(self.temporal_ply_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{frame:05d}.ply"
        tensors = self._gaussian_ply_tensors()
        count = tensors["x"].shape[0]
        ExportGaussianSplat.write_ply(str(path), count, tensors)
        CONSOLE.log(f"[temporal] saved frame {frame + 1}/{self.num_frames}: {path} ({count} gaussians).")
