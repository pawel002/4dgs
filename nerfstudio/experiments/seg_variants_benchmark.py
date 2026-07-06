#!/usr/bin/env python
"""Prototype benchmark for *improved* model-free segmentation variants.

The production background subtraction (``TemporalDataParser._apply_bg_subtraction``)
thresholds the mean absolute RGB difference against the empty-room plate at a single
global threshold. Its residual failure is *camouflage*: where the subject's colours
locally match the background (the mannequin over the rug), the difference is weak and
the silhouette develops holes — but the difference is weak, not zero. A single global
threshold cannot admit that weak evidence without also admitting shadows and
indirect-light flicker, which live at the same magnitudes.

This script benchmarks variants that use *more of the evidence* while staying
model-free:

* ``mean``       — the production difference, d = mean_k |I_k - B_k| (baseline).
* ``max``        — d = max_k |I_k - B_k|: a strong single-channel change is not
                   diluted by the two unchanged channels.
* ``hyst``       — hysteresis (two thresholds): pixels with d > tau_hi are kept
                   unconditionally (seeds); pixels with tau_lo < d <= tau_hi are kept
                   only if geodesically CONNECTED to a seed. Camouflaged body regions
                   are weak evidence attached to strong evidence; shadow fringe is
                   weak evidence attached to nothing (after the seeds are cleaned).
* ``hyst+shadow``— hysteresis where the weak band additionally excludes pixels that
                   *darkened without changing chromaticity* (the photometric signature
                   of a cast shadow), so the weak band cannot leak through the contact
                   shadow under the feet.

All variants share the production morphology (opening -> dilation -> final erosions).
Measured per (camera, frame): mask IoU / precision / recall vs the ground-truth
silhouettes (`output/black-bg-frames`). Sweeps run on a frame subset.

Run (GPU recommended, the geodesic reconstruction is iterative)::

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/seg_variants_benchmark.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image


@dataclass
class Config:
    dataset: Path = Path("../output/dataset")
    gt: Path = Path("../output/black-bg-frames")
    cameras: str = "static1,static2,static3,static4"
    num_frames: int = 150
    stride: int = 10
    """Sweep uses every Nth frame (plus a dense early-frame window, see below)."""
    early_window: int = 20
    """Also evaluate frames 1..early_window densely — the camouflage failure lives there."""
    gt_threshold: float = 0.02
    open_radius: int = 1
    dilate: int = 2
    erode: int = 2
    device: str = "cuda"
    output_dir: Path = Path("experiments/outputs/bgsub")


def load_rgb(path: Path, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0).to(device)


def morphology(mask: torch.Tensor, open_r: int, dil: int, erode: int) -> torch.Tensor:
    """The production cleanup: opening -> dilation -> N final erosions. mask [H,W] float."""
    m = mask[None, None]
    if open_r > 0:
        m = -F.max_pool2d(-m, kernel_size=2 * open_r + 1, stride=1, padding=open_r)
        m = F.max_pool2d(m, kernel_size=2 * open_r + 1, stride=1, padding=open_r)
    if dil > 0:
        m = F.max_pool2d(m, kernel_size=2 * dil + 1, stride=1, padding=dil)
    for _ in range(erode):
        m = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
    return m[0, 0]


def geodesic_reconstruct(seed: torch.Tensor, allowed: torch.Tensor, radius: int = 2, iters: int = 200) -> torch.Tensor:
    """Binary reconstruction-by-dilation of `seed` constrained to `allowed` (both [H,W] float 0/1).

    Repeats (dilate seed, intersect with allowed) until fixpoint. This is what makes
    hysteresis *connectivity-aware*: weak pixels join only if a path of weak pixels
    connects them to a strong seed.
    """
    cur = (seed * allowed)[None, None]
    allowed = allowed[None, None]
    k = 2 * radius + 1
    for _ in range(iters):
        grown = F.max_pool2d(cur, kernel_size=k, stride=1, padding=radius) * allowed
        if bool((grown == cur).all()):
            break
        cur = grown
    return cur[0, 0]


def shadow_free_band(img: torch.Tensor, plate: torch.Tensor, chroma_thr: float = 0.04) -> torch.Tensor:
    """Boolean [H,W]: pixels whose change is NOT explainable as a cast shadow.

    A cast shadow scales the illumination down: luminance drops, chromaticity
    (colour direction) is nearly unchanged. A genuinely different surface changes
    chromaticity (or gets brighter). Keep a pixel when its chromaticity moved, or it
    brightened.
    """
    lum_i = img.mean(-1)
    lum_p = plate.mean(-1)
    chr_i = img / lum_i.clamp_min(1e-4)[..., None]
    chr_p = plate / lum_p.clamp_min(1e-4)[..., None]
    chroma_change = (chr_i - chr_p).abs().mean(-1)
    darkened = lum_i < lum_p
    return (chroma_change > chroma_thr) | (~darkened)


def make_mask(
    variant: str,
    img: torch.Tensor,
    plate: torch.Tensor,
    thr: float,
    thr_lo: float,
    cfg: Config,
    chroma_thr: float = 0.04,
) -> torch.Tensor:
    diff_mean = (img - plate).abs().mean(dim=-1)
    if variant == "mean":
        raw = (diff_mean > thr).float()
    elif variant == "max":
        raw = ((img - plate).abs().amax(dim=-1) > thr).float()
    elif variant in ("hyst", "hyst+shadow"):
        strong = (diff_mean > thr).float()
        # Clean the seeds first so isolated speckle cannot recruit a weak region.
        strong = morphology(strong, cfg.open_radius, 0, 0)
        weak = diff_mean > thr_lo
        if variant == "hyst+shadow":
            weak = weak & shadow_free_band(img, plate, chroma_thr)
        raw = geodesic_reconstruct(strong, weak.float())
    elif variant == "shadow-noconn":
        # Ablation: the same two-band evidence but WITHOUT the connectivity constraint —
        # every shadow-free weak pixel is admitted, connected to the subject or not.
        strong = diff_mean > thr
        weak = (diff_mean > thr_lo) & shadow_free_band(img, plate, chroma_thr)
        raw = (strong | weak).float()
    else:
        raise ValueError(variant)
    return morphology(raw, cfg.open_radius, cfg.dilate, cfg.erode)


def mask_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    p, g = pred > 0.5, gt
    inter = (p & g).sum().item()
    union = (p | g).sum().item()
    return {
        "iou": inter / max(union, 1),
        "precision": inter / max(p.sum().item(), 1),
        "recall": inter / max(g.sum().item(), 1),
    }


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cams = cfg.cameras.split(",")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    plates = {c: load_rgb(cfg.dataset / c / "images" / "0000.png", device) for c in cams}

    frames = sorted(set(range(1, cfg.early_window + 1)) | set(range(1, cfg.num_frames + 1, cfg.stride)))
    print(f"[seg-variants] {len(frames)} frames x {len(cams)} cameras on {device}")

    # (variant, thr, thr_lo) sweep grid. thr for hyst is tau_hi.
    grid = []
    for thr in (0.04, 0.05, 0.06, 0.08, 0.1):
        grid.append(("mean", thr, 0.0, 0.04))
    for thr in (0.06, 0.08, 0.1, 0.12, 0.15):
        grid.append(("max", thr, 0.0, 0.04))
    for thr_hi in (0.08, 0.1, 0.12):
        for thr_lo in (0.01, 0.02, 0.03):
            grid.append(("hyst", thr_hi, thr_lo, 0.04))
            grid.append(("hyst+shadow", thr_hi, thr_lo, 0.04))
    # Connectivity ablation + chroma-threshold sensitivity at the winning operating point.
    for thr_lo in (0.01, 0.02, 0.03):
        grid.append(("shadow-noconn", 0.1, thr_lo, 0.04))
    for chroma in (0.02, 0.06, 0.08, 0.12):
        grid.append(("hyst+shadow", 0.1, 0.01, chroma))

    results: List[Dict] = []
    # Cache images once.
    imgs = {(c, t): load_rgb(cfg.dataset / c / "images" / f"{t:04d}.png", device) for c in cams for t in frames}
    gts = {
        (c, t): load_rgb(cfg.gt / c / "images" / f"{t:04d}.png", device).mean(-1) > cfg.gt_threshold
        for c in cams
        for t in frames
    }

    for variant, thr, thr_lo, chroma in grid:
        per_frame: Dict[int, List[Dict[str, float]]] = {t: [] for t in frames}
        for t in frames:
            for c in cams:
                mask = make_mask(variant, imgs[(c, t)], plates[c], thr, thr_lo, cfg, chroma)
                per_frame[t].append(mask_metrics(mask, gts[(c, t)]))
        allm = [m for t in frames for m in per_frame[t]]
        early = [m for t in frames if t <= cfg.early_window for m in per_frame[t]]
        rec = {
            "variant": variant,
            "thr": thr,
            "thr_lo": thr_lo,
            "chroma_thr": chroma,
            **{k: float(np.mean([m[k] for m in allm])) for k in ("iou", "precision", "recall")},
            **{f"early_{k}": float(np.mean([m[k] for m in early])) for k in ("iou", "precision", "recall")},
            "per_frame_iou": {str(t): float(np.mean([m["iou"] for m in per_frame[t]])) for t in frames},
        }
        results.append(rec)
        print(
            f"[{variant:13s}] thr={thr:.3f} lo={thr_lo:.3f} chr={chroma:.2f} | IoU={rec['iou']:.4f} "
            f"P={rec['precision']:.4f} R={rec['recall']:.4f} | early IoU={rec['early_iou']:.4f}"
        )

    out_path = cfg.output_dir / "seg_variants.json"
    out_path.write_text(json.dumps({"config": {k: str(v) for k, v in vars(cfg).items()}, "results": results}, indent=2))
    print(f"saved {out_path}")

    best = max(results, key=lambda r: r["iou"])
    print(f"BEST overall IoU: {best['variant']} thr={best['thr']} lo={best['thr_lo']} IoU={best['iou']:.4f}")
    best_e = max(results, key=lambda r: r["early_iou"])
    print(f"BEST early  IoU: {best_e['variant']} thr={best_e['thr']} lo={best_e['thr_lo']} IoU={best_e['early_iou']:.4f}")


if __name__ == "__main__":
    main(tyro.cli(Config))
