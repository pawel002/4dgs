#!/usr/bin/env python
"""Qualitative thesis figures: segmentation montage, GT-vs-render montage, merged
background+foreground composites, and per-primitive trajectory/displacement plots.

Run (needs a GPU for the render-based figures)::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.make_qualitative \
        --run-dir splats/thesis/full150/temporal-splatfacto/thesis \
        --background-run splats/thesis/background/splatfacto/thesis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image

from experiments.eval_temporal import build_eval_cameras, load_gaussians, render_gaussians
from experiments.temporal_compress import read_ply

OUT = Path("../artefacts/thesis/res/03_research/results")
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name: str, dpi=220):
    OUT.mkdir(parents=True, exist_ok=True)
    p = (OUT / name).with_suffix(".png")
    fig.savefig(p, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[saved] {p}")


def load_rgb_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def crop_bbox(mask: np.ndarray, margin: float = 0.35, square=False) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    my, mx = int(h * margin), int(w * margin)
    y0, y1 = max(0, y0 - my), min(mask.shape[0], y1 + my)
    x0, x1 = max(0, x0 - mx), min(mask.shape[1], x1 + mx)
    return y0, y1, x0, x1


# ------------------------------------------------------------------ 1. bg-sub montage


def fig_bgsub_montage(dataset: Path, gt_dir: Path, cam: str, frame: int,
                      threshold: float = 0.06, erode: int = 2,
                      bgsub_dir: str = "images_bgsub_t06e2"):
    img = load_rgb_np(dataset / cam / "images" / f"{frame:04d}.png")
    plate = load_rgb_np(dataset / cam / "images" / "0000.png")
    gt = load_rgb_np(gt_dir / cam / "images" / f"{frame:04d}.png")
    seg = load_rgb_np(dataset / cam / bgsub_dir / f"{frame:04d}.png")

    diff = np.abs(img - plate).mean(-1)
    m = torch.from_numpy((diff > threshold).astype(np.float32))[None, None]
    m = -F.max_pool2d(-m, 3, 1, 1)
    m = F.max_pool2d(m, 3, 1, 1)
    m = F.max_pool2d(m, 5, 1, 2)
    for _ in range(erode):
        m = -F.max_pool2d(-m, 3, 1, 1)
    mask = m[0, 0].numpy()

    gt_mask = gt.mean(-1) > 0.02
    y0, y1, x0, x1 = crop_bbox(gt_mask | (mask > 0.5), margin=0.45)

    panels = [
        (img[y0:y1, x0:x1], "input frame $I_{c,t}$", None),
        (plate[y0:y1, x0:x1], "background plate $I^{\\mathrm{bg}}_c$", None),
        (diff[y0:y1, x0:x1], "change magnitude $d_{c,t}$", "magma"),
        (mask[y0:y1, x0:x1], "silhouette $\\mathcal{S}_{c,t}$", "gray"),
        (seg[y0:y1, x0:x1], "segmented $\\tilde I_{c,t}$", None),
        (gt[y0:y1, x0:x1], "ground truth", None),
    ]
    n = len(panels)
    tile_w = 6.3 / n
    fig, axes = plt.subplots(1, n, figsize=(6.3, tile_w * (y1 - y0) / (x1 - x0) + 0.42))
    for ax, (im, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=6.5)
        ax.axis("off")
    fig.tight_layout(pad=0.25)
    _save(fig, "bgsub_montage.png", dpi=300)


# ------------------------------------------------------------------ 2. GT vs render montage


def fig_temporal_montage(run_dir: Path, data: Path, gt_dir: Path, cam_idx: int, frames: List[int],
                         device="cuda"):
    cams, _ = build_eval_cameras(data, eval_scale=1.0)
    cam = cams[cam_idx]
    gts, renders, masks = [], [], []
    for f in frames:
        g = load_gaussians(run_dir / "temporal_frames" / f"{f:05d}.ply", device)
        rgb, _ = render_gaussians(g, cam["c2w"], cam["K"], cam["width"], cam["height"])
        renders.append(rgb.cpu().numpy())
        gt = load_rgb_np(gt_dir / cam["name"] / "images" / f"{f + 1:04d}.png")
        gts.append(gt)
        masks.append(gt.mean(-1) > 0.02)

    # per-frame crops with one common tile size, centred on each frame's silhouette
    boxes = [crop_bbox(m, margin=0.3) for m in masks]
    th = max(y1 - y0 for y0, y1, _, _ in boxes)
    tw = max(x1 - x0 for _, _, x0, x1 in boxes)
    H, W = masks[0].shape

    def tile(im, box):
        y0, y1, x0, x1 = box
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        y0 = int(np.clip(cy - th // 2, 0, H - th)); x0 = int(np.clip(cx - tw // 2, 0, W - tw))
        return im[y0:y0 + th, x0:x0 + tw]

    n = len(frames)
    ar = th / tw
    fig, axes = plt.subplots(2, n, figsize=(6.3, 2 * ar * 6.3 / n + 0.55))
    for j, f in enumerate(frames):
        axes[0, j].imshow(tile(gts[j], boxes[j]))
        axes[0, j].set_title(f"$t = {f}$", fontsize=9)
        axes[1, j].imshow(np.clip(tile(renders[j], boxes[j]), 0, 1))
        for i in (0, 1):
            axes[i, j].axis("off")
    axes[0, 0].text(-0.09, 0.5, "ground truth", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=9)
    axes[1, 0].text(-0.09, 0.5, "reconstruction", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=9)
    fig.tight_layout(pad=0.3)
    _save(fig, "temporal_montage.png")


# ------------------------------------------------------------------ 3. composite


def _load_dp_transform(run_dir: Path) -> Tuple[np.ndarray, float]:
    d = json.loads((run_dir / "dataparser_transforms.json").read_text())
    T = np.array(d["transform"], dtype=np.float64)  # [3,4]
    return T, float(d["scale"])


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def load_background_in_fg_frame(bg_ply: Path, bg_run: Path, fg_run: Path, device="cuda"):
    """Load the splatfacto background export and re-express it in the temporal run's
    normalized frame (each run's dataparser normalizes poses differently)."""
    g = load_gaussians(bg_ply, device)
    Tb, sb = _load_dp_transform(bg_run)
    Tf, sf = _load_dp_transform(fg_run)
    Rb, tb = Tb[:, :3], Tb[:, 3]
    Rf, tf = Tf[:, :3], Tf[:, 3]
    R_rel = Rf @ Rb.T
    t_rel = sf * (tf - R_rel @ tb * 1.0)  # combined below explicitly
    ang = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    print(f"[composite] relative rotation between frames: {ang:.2f} deg, scale x{sf/sb:.4f}")

    means = g["means"].cpu().numpy().astype(np.float64)
    # bg-norm -> world -> fg-norm
    world = (means / sb - tb) @ Rb  # (R_b^T (p/s - t)) — row-vector form
    fg = (world @ Rf.T + tf) * sf
    g["means"] = torch.from_numpy(fg.astype(np.float32)).to(device)
    g["scales"] = g["scales"] + float(np.log(sf / sb))
    q_rel = _rotmat_to_quat(R_rel)
    quats = g["quats"].cpu().numpy().astype(np.float64)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    g["quats"] = torch.from_numpy(_quat_mul(q_rel[None], quats).astype(np.float32)).to(device)
    return g


def merge_gaussians(a: Dict, b: Dict) -> Dict:
    """Concatenate two gaussian sets (padding SH rest bands to the larger degree)."""
    ka, kb = a["colors"].shape[1], b["colors"].shape[1]
    k = max(ka, kb)
    def pad(c):
        if c.shape[1] < k:
            z = torch.zeros(c.shape[0], k - c.shape[1], 3, device=c.device)
            c = torch.cat([c, z], dim=1)
        return c
    return {
        "means": torch.cat([a["means"], b["means"]]),
        "scales": torch.cat([a["scales"], b["scales"]]),
        "quats": torch.cat([a["quats"], b["quats"]]),
        "opacities": torch.cat([a["opacities"], b["opacities"]]),
        "colors": torch.cat([pad(a["colors"]), pad(b["colors"])]),
        "sh_degree": int(round(np.sqrt(k))) - 1,
    }


def fig_composite(run_dir: Path, bg_ply: Path, bg_run: Path, data: Path, frames: List[int],
                  cam_idx: int, device="cuda"):
    cams, _ = build_eval_cameras(data, eval_scale=1.0)
    cam = cams[cam_idx]
    bg = load_background_in_fg_frame(bg_ply, bg_run, run_dir, device)

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(6.3, 2 * 6.3 / n * 9 / 16 + 0.5))
    for j, f in enumerate(frames):
        raw = load_rgb_np(data / cam["name"] / "images" / f"{f + 1:04d}.png")
        fg = load_gaussians(run_dir / "temporal_frames" / f"{f:05d}.ply", device)
        both = merge_gaussians(bg, fg)
        rgb, _ = render_gaussians(both, cam["c2w"], cam["K"], cam["width"], cam["height"])
        axes[0, j].imshow(raw)
        axes[0, j].set_title(f"$t = {f}$", fontsize=9)
        axes[1, j].imshow(np.clip(rgb.cpu().numpy(), 0, 1))
        for i in (0, 1):
            axes[i, j].axis("off")
    axes[0, 0].text(-0.07, 0.5, "input frame", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=9)
    axes[1, 0].text(-0.07, 0.5, "merged render", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=9)
    fig.tight_layout(pad=0.3)
    _save(fig, "composite_montage.png")


# ------------------------------------------------------------------ 4. trajectories


def fig_trajectories(run_dir: Path, n_show: int = 400, seed: int = 0):
    plys = sorted((run_dir / "temporal_frames").glob("*.ply"))
    means = []
    for p in plys:
        props, data = read_ply(str(p))
        ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
        means.append(data[:, [ix, iy, iz]])
    M = np.stack(means)  # [F, N, 3]
    F_, N = M.shape[:2]
    D = np.linalg.norm(np.diff(M, axis=0), axis=-1).sum(axis=0)  # [N]

    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(n_show, N), replace=False)

    # pick the two world axes with the largest motion extent for the 2D projection
    extent = (M.max(axis=(0, 1)) - M.min(axis=(0, 1)))
    a1, a2 = np.argsort(extent)[::-1][:2]
    if extent[2] > 0.3 * extent[a2]:
        a2 = 2  # prefer the vertical axis as the second one when it carries any motion
    names = ["x", "y", "z"]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    t_colors = plt.cm.viridis(np.linspace(0, 1, F_))
    for i in idx:
        ax.plot(M[:, i, a1], M[:, i, a2], color=GRAY, alpha=0.06, linewidth=0.5, rasterized=True)
    for f in np.linspace(0, F_ - 1, 6).astype(int):
        ax.scatter(M[f, idx, a1], M[f, idx, a2], s=1.6, color=t_colors[f], rasterized=True)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, F_ - 1))
    fig.colorbar(sm, ax=ax, label="frame $t$", fraction=0.05, pad=0.03)
    ax.set_xlabel(f"${names[a1]}$")
    ax.set_ylabel(f"${names[a2]}$")
    ax.set_title("(a) per-primitive trajectories", pad=8)
    ax.grid(True, color="#e2e2e2", linewidth=0.5)

    ax = axes[1]
    ax.hist(D, bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("cumulative displacement $D_i$")
    ax.set_ylabel("number of primitives")
    ax.set_title("(b) displacement distribution", pad=8)
    ax.grid(True, axis="y", color="#e2e2e2", linewidth=0.5)
    _save(fig, "trajectories.pdf", dpi=300)


# ------------------------------------------------------------------ main


@dataclass
class Config:
    run_dir: Path = Path("splats/thesis/full150/temporal-splatfacto/thesis")
    background_run: Optional[Path] = None
    background_ply: Optional[Path] = None
    data: Path = Path("../output/dataset")
    gt: Path = Path("../output/black-bg-frames")
    bgsub_cam: str = "static2"
    bgsub_frame: int = 75
    montage_cam: int = 0
    montage_frames: str = "0,30,60,90,120,149"
    composite_frames: str = "10,75,140"
    composite_cam: int = 1
    only: str = ""
    """Comma-separated subset of {bgsub,montage,composite,traj}; empty = all available."""


def main(cfg: Config) -> None:
    which = set(cfg.only.split(",")) if cfg.only else {"bgsub", "montage", "composite", "traj"}
    if "bgsub" in which:
        fig_bgsub_montage(cfg.data, cfg.gt, cfg.bgsub_cam, cfg.bgsub_frame)
    if "traj" in which:
        fig_trajectories(cfg.run_dir)
    if "montage" in which:
        fig_temporal_montage(cfg.run_dir, cfg.data, cfg.gt, cfg.montage_cam,
                             [int(x) for x in cfg.montage_frames.split(",")])
    if "composite" in which:
        if cfg.background_run is None or cfg.background_ply is None:
            print("[skip] composite: --background-run/--background-ply not given")
        else:
            fig_composite(cfg.run_dir, cfg.background_ply, cfg.background_run, cfg.data,
                          [int(x) for x in cfg.composite_frames.split(",")], cfg.composite_cam)


if __name__ == "__main__":
    main(tyro.cli(Config))
