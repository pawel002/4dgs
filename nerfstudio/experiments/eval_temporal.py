#!/usr/bin/env python
"""Offline evaluation of a temporal-splatfacto run against ground-truth renders.

Loads the per-frame gaussians (``temporal_frames/*.ply``), re-renders every frame from
the four static training cameras with gsplat, and compares against the *true*
black-background renders in ``output/black-bg-frames`` (not the background-subtracted
training inputs), so the numbers measure end-to-end quality of the whole pipeline
including segmentation error.

Per frame (averaged over the 4 cameras) it records:
  * object PSNR   — PSNR over the GT-silhouette pixels only,
  * full PSNR     — over the whole image,
  * object SSIM   — SSIM on the bounding box of the GT silhouette,
  * coverage      — mean rendered alpha inside the eroded GT silhouette,
  * leak          — mean rendered alpha outside the dilated GT silhouette,
  * cloud stats   — gaussian count, RMS/mean displacement vs previous frame,
                    95th/100th percentile distance from the cloud centroid,
                    mean |f_dc - f_dc(frame 0)| (colour drift).

Run::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/eval_temporal.py \
        --run-dir splats/thesis/abl_default/temporal-splatfacto/thesis \
        --output experiments/outputs/eval/abl_default.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image

from gsplat.rendering import rasterization

from experiments.temporal_compress import read_ply
from nerfstudio.data.dataparsers.temporal_dataparser import TemporalDataParserConfig
from nerfstudio.models.splatfacto import get_viewmat


# ---------------------------------------------------------------- ply -> gaussians


def load_gaussians(ply_path: Path, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Read a temporal frame ply into raw (pre-activation) gaussian tensors."""
    props, data = read_ply(str(ply_path))
    col = {p: torch.from_numpy(np.ascontiguousarray(data[:, i])) for i, p in enumerate(props)}
    n = data.shape[0]
    means = torch.stack([col["x"], col["y"], col["z"]], dim=-1)
    scales = torch.stack([col[f"scale_{i}"] for i in range(3)], dim=-1)
    quats = torch.stack([col[f"rot_{i}"] for i in range(4)], dim=-1)
    opacities = col["opacity"]
    f_dc = torch.stack([col[f"f_dc_{i}"] for i in range(3)], dim=-1)  # [N, 3]
    rest_names = sorted((p for p in props if p.startswith("f_rest_")), key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        rest = torch.stack([col[p] for p in rest_names], dim=-1)  # [N, 3*Kr] channel-major
        kr = rest.shape[-1] // 3
        f_rest = rest.reshape(n, 3, kr).transpose(1, 2)  # [N, Kr, 3]
        colors = torch.cat([f_dc[:, None, :], f_rest], dim=1)  # [N, 1+Kr, 3]
    else:
        colors = f_dc[:, None, :]
    sh_degree = int(round(np.sqrt(colors.shape[1]))) - 1
    out = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": sh_degree,
    }
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in out.items()}


@torch.no_grad()
def render_gaussians(
    g: Dict[str, torch.Tensor],
    c2w: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    background: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize raw gaussians exactly like splatfacto. Returns (rgb [H,W,3], alpha [H,W])."""
    device = g["means"].device
    viewmat = get_viewmat(c2w[None].to(device))
    render, alpha, _ = rasterization(
        means=g["means"],
        quats=g["quats"],
        scales=torch.exp(g["scales"]),
        opacities=torch.sigmoid(g["opacities"]),
        colors=g["colors"],
        viewmats=viewmat,
        Ks=K[None].to(device),
        width=width,
        height=height,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=g["sh_degree"],
        sparse_grad=False,
        absgrad=False,
        rasterize_mode="classic",
    )
    alpha = alpha[0, ..., 0]
    rgb = render[0, ..., :3]
    if background is None:
        background = torch.zeros(3, device=device)
    rgb = (rgb + (1.0 - alpha[..., None]) * background).clamp(0.0, 1.0)
    return rgb, alpha


# ---------------------------------------------------------------- cameras & GT


def build_eval_cameras(data: Path, eval_scale: float):
    """Frame-0 poses/intrinsics of the 4 static cameras (poses are constant over time),
    in the same normalized frame the plys were trained (and saved) in."""
    dp = TemporalDataParserConfig(data=data, init_visual_hull=False).setup()
    out = dp._generate_dataparser_outputs("train")
    md = out.metadata
    frame_idx = np.asarray(md["frame_indices"])
    cam_idx = np.asarray(md["camera_indices"])
    names = md["camera_names"]
    sel = np.nonzero(frame_idx == 0)[0]
    sel = sel[np.argsort(cam_idx[sel])]
    cams = []
    for i in sel:
        c = out.cameras[int(i)]
        fx, fy = float(c.fx) * eval_scale, float(c.fy) * eval_scale
        cx, cy = float(c.cx) * eval_scale, float(c.cy) * eval_scale
        w, h = int(round(int(c.width) * eval_scale)), int(round(int(c.height) * eval_scale))
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        cams.append({"c2w": c.camera_to_worlds.clone(), "K": K, "width": w, "height": h,
                     "name": names[int(cam_idx[i])]})
    return cams, int(md["num_frames"])


def load_gt(gt_dir: Path, cam_name: str, frame: int, width: int, height: int, device: str) -> torch.Tensor:
    img = Image.open(gt_dir / cam_name / "images" / f"{frame + 1:04d}.png").convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).to(device)


# ---------------------------------------------------------------- metrics


def _pool_mask(mask: torch.Tensor, r: int, erode: bool) -> torch.Tensor:
    m = mask.float()[None, None]
    if erode:
        m = -F.max_pool2d(-m, kernel_size=2 * r + 1, stride=1, padding=r)
    else:
        m = F.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r)
    return m[0, 0] > 0.5


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    diff = (pred - gt) if mask is None else (pred - gt)[mask]
    if diff.numel() == 0:
        return float("nan")
    mse = (diff**2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def bbox_ssim(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """SSIM on the tight bounding box of the GT silhouette (padded by 8 px)."""
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return float("nan")
    pad = 8
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, mask.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, mask.shape[1])
    p = pred[y0:y1, x0:x1].permute(2, 0, 1)[None]
    g = gt[y0:y1, x0:x1].permute(2, 0, 1)[None]
    return float(ssim(p, g, data_range=1.0))


# ---------------------------------------------------------------- main


@dataclass
class Config:
    run_dir: Path
    """Run directory (contains temporal_frames/) or the temporal_frames dir itself."""
    data: Path = Path("../output/dataset")
    gt: Path = Path("../output/black-bg-frames")
    output: Path = Path("experiments/outputs/eval/run.json")
    eval_scale: float = 0.5
    gt_threshold: float = 0.02
    """GT pixel is object if its mean RGB is above this."""
    core_erode: int = 3
    """Erosion (px) defining the object core for the coverage metric."""
    leak_dilate: int = 5
    """Dilation (px); everything outside it is 'clearly background' for the leak metric."""
    max_frames: int = -1
    device: str = "cuda"


def main(cfg: Config) -> None:
    frames_dir = cfg.run_dir if cfg.run_dir.name == "temporal_frames" else cfg.run_dir / "temporal_frames"
    plys = sorted(frames_dir.glob("*.ply"))
    assert plys, f"no plys found in {frames_dir}"
    if cfg.max_frames > 0:
        plys = plys[: cfg.max_frames]

    cams, _ = build_eval_cameras(cfg.data, cfg.eval_scale)
    gt_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    records: List[Dict] = []
    means0_colors: Optional[torch.Tensor] = None
    prev_means: Optional[torch.Tensor] = None
    for f, ply in enumerate(plys):
        g = load_gaussians(ply, cfg.device)
        if means0_colors is None:
            means0_colors = g["colors"][:, 0, :].clone()

        rec: Dict[str, float] = {"frame": f, "num_gaussians": int(g["means"].shape[0])}
        obj_psnr, full_psnr, obj_ssim, coverage, leak = [], [], [], [], []
        for ci, cam in enumerate(cams):
            rgb, alpha = render_gaussians(g, cam["c2w"], cam["K"], cam["width"], cam["height"])
            gt = load_gt(cfg.gt, cam["name"], f, cam["width"], cam["height"], cfg.device)
            gt_mask = gt.mean(-1) > cfg.gt_threshold
            core = _pool_mask(gt_mask, cfg.core_erode, erode=True)
            outside = ~_pool_mask(gt_mask, cfg.leak_dilate, erode=False)
            obj_psnr.append(masked_psnr(rgb, gt, gt_mask))
            full_psnr.append(masked_psnr(rgb, gt, None))
            obj_ssim.append(bbox_ssim(rgb, gt, gt_mask))
            coverage.append(float(alpha[core].mean()) if core.any() else float("nan"))
            leak.append(float(alpha[outside].mean()) if outside.any() else float("nan"))
        rec["object_psnr"] = float(np.nanmean(obj_psnr))
        rec["full_psnr"] = float(np.nanmean(full_psnr))
        rec["object_ssim"] = float(np.nanmean(obj_ssim))
        rec["coverage"] = float(np.nanmean(coverage))
        rec["leak"] = float(np.nanmean(leak))
        rec["per_camera_object_psnr"] = obj_psnr

        centroid = g["means"].median(dim=0).values
        d = (g["means"] - centroid).norm(dim=-1)
        rec["radius_p95"] = float(torch.quantile(d, 0.95))
        rec["radius_max"] = float(d.max())
        if prev_means is not None and prev_means.shape == g["means"].shape:
            step = (g["means"] - prev_means).norm(dim=-1)
            rec["mean_displacement"] = float(step.mean())
            rec["max_displacement"] = float(step.max())
        if means0_colors.shape == g["colors"][:, 0, :].shape:
            rec["color_drift"] = float((g["colors"][:, 0, :] - means0_colors).abs().mean())
        prev_means = g["means"].clone()
        records.append(rec)
        if f % 25 == 0 or f == len(plys) - 1:
            print(f"[eval] frame {f}: objPSNR={rec['object_psnr']:.2f} cov={rec['coverage']:.3f} "
                  f"leak={rec['leak']:.5f}")

    def series(key: str) -> List[float]:
        return [r.get(key, float("nan")) for r in records]

    out = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "frames": records,
        "summary": {
            k: {"mean": float(np.nanmean(series(k))), "first": series(k)[0], "last": series(k)[-1]}
            for k in ("object_psnr", "full_psnr", "object_ssim", "coverage", "leak",
                      "radius_p95", "radius_max", "color_drift")
        },
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(out, indent=2))
    print(f"saved {cfg.output}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
