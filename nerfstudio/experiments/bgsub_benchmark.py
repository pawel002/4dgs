#!/usr/bin/env python
"""Benchmark the model-free background subtraction against ground-truth renders.

The dataset ``output/dataset/static*`` holds un-segmented room frames plus an
empty-room plate (``0000.png``); ``output/black-bg-frames/static*`` holds the *same*
subject frames rendered by Blender with the room hidden (black background) — an exact
ground truth for what the segmentation should recover.

Measured per camera and frame:
  * mask IoU / precision / recall of the predicted silhouette vs the GT silhouette,
  * PSNR of the segmented image vs the GT black-background render (whole image).

Plus a threshold sweep (with and without the morphological cleanup) on a frame subset.

Run (CPU only)::

    .venv/bin/python experiments/bgsub_benchmark.py
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
    threshold: float = 0.06         # default pipeline threshold (tuned by the joint sweep)
    open_radius: int = 1            # default morphological opening
    dilate: int = 2                 # default final dilation
    erode: int = 2                  # final 3x3 erosion steps after the dilation (tuned)
    gt_threshold: float = 0.02      # GT pixel is object if mean RGB above this
    sweep_thresholds: str = "0.02,0.05,0.075,0.1,0.15,0.2,0.3"
    sweep_erosions: str = "0,1,2,3,4,5"
    sweep_stride: int = 10          # sweep uses every Nth frame
    output_dir: Path = Path("experiments/outputs/bgsub")


def load_rgb(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0)


def segment_mask(img: torch.Tensor, plate: torch.Tensor, thr: float, open_r: int, dil: int,
                 erode: int = 0) -> torch.Tensor:
    """Replicates TemporalDataParser._apply_bg_subtraction, returning the binary mask [H,W]."""
    diff = (img - plate).abs().mean(dim=-1)
    mask = (diff > thr).float()[None, None]
    if open_r > 0:
        mask = -F.max_pool2d(-mask, kernel_size=2 * open_r + 1, stride=1, padding=open_r)
        mask = F.max_pool2d(mask, kernel_size=2 * open_r + 1, stride=1, padding=open_r)
    if dil > 0:
        mask = F.max_pool2d(mask, kernel_size=2 * dil + 1, stride=1, padding=dil)
    for _ in range(erode):
        mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return mask[0, 0]


def mask_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    p, g = pred.bool(), gt.bool()
    inter = (p & g).sum().item()
    union = (p | g).sum().item()
    return {
        "iou": inter / max(union, 1),
        "precision": inter / max(p.sum().item(), 1),
        "recall": inter / max(g.sum().item(), 1),
    }


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cams = cfg.cameras.split(",")
    plates = {c: load_rgb(cfg.dataset / c / "images" / "0000.png") for c in cams}

    # ---- default-parameter evaluation over every frame ----
    per_frame: Dict[str, List[Dict[str, float]]] = {c: [] for c in cams}
    for t in range(1, cfg.num_frames + 1):
        for c in cams:
            img = load_rgb(cfg.dataset / c / "images" / f"{t:04d}.png")
            gt_img = load_rgb(cfg.gt / c / "images" / f"{t:04d}.png")
            gt_mask = gt_img.mean(-1) > cfg.gt_threshold
            mask = segment_mask(img, plates[c], cfg.threshold, cfg.open_radius, cfg.dilate,
                                cfg.erode)
            seg = img * mask[..., None]
            m = mask_metrics(mask, gt_mask)
            m["psnr"] = psnr(seg, gt_img)
            per_frame[c].append(m)
        if t % 25 == 0:
            print(f"[default eval] frame {t}/{cfg.num_frames}")

    def agg(key: str) -> Dict[str, float]:
        vals = np.array([[m[key] for m in per_frame[c]] for c in cams])
        return {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}

    summary = {k: agg(k) for k in ("iou", "precision", "recall", "psnr")}
    print("default params:", json.dumps(summary, indent=2))

    # ---- threshold sweep (subset of frames), with and without morphology ----
    thresholds = [float(x) for x in cfg.sweep_thresholds.split(",")]
    frames = list(range(1, cfg.num_frames + 1, cfg.sweep_stride))
    sweep = {"thresholds": thresholds, "with_morph": [], "no_morph": []}
    for thr in thresholds:
        for key, (o, d, e) in (("with_morph", (cfg.open_radius, cfg.dilate, cfg.erode)),
                               ("no_morph", (0, 0, 0))):
            ms = []
            for t in frames:
                for c in cams:
                    img = load_rgb(cfg.dataset / c / "images" / f"{t:04d}.png")
                    gt_img = load_rgb(cfg.gt / c / "images" / f"{t:04d}.png")
                    gt_mask = gt_img.mean(-1) > cfg.gt_threshold
                    mask = segment_mask(img, plates[c], thr, o, d, e)
                    ms.append(mask_metrics(mask, gt_mask))
            sweep[key].append({k: float(np.mean([m[k] for m in ms])) for k in ("iou", "precision", "recall")})
        print(f"[sweep] thr={thr}: IoU with_morph={sweep['with_morph'][-1]['iou']:.4f} "
              f"no_morph={sweep['no_morph'][-1]['iou']:.4f}")

    # ---- erosion sweep (subset of frames) at the default threshold ----
    erosions = [int(x) for x in cfg.sweep_erosions.split(",")]
    erosion_sweep = {"erosions": erosions, "metrics": []}
    for e in erosions:
        ms = []
        for t in frames:
            for c in cams:
                img = load_rgb(cfg.dataset / c / "images" / f"{t:04d}.png")
                gt_img = load_rgb(cfg.gt / c / "images" / f"{t:04d}.png")
                gt_mask = gt_img.mean(-1) > cfg.gt_threshold
                mask = segment_mask(img, plates[c], cfg.threshold, cfg.open_radius,
                                    cfg.dilate, e)
                ms.append(mask_metrics(mask, gt_mask))
        erosion_sweep["metrics"].append(
            {k: float(np.mean([m[k] for m in ms])) for k in ("iou", "precision", "recall")}
        )
        m = erosion_sweep["metrics"][-1]
        print(f"[erosion sweep] erode={e}: IoU={m['iou']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

    out = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "summary_default": summary,
        "per_frame": {c: per_frame[c] for c in cams},
        "sweep": sweep,
        "erosion_sweep": erosion_sweep,
    }
    (cfg.output_dir / "bgsub_benchmark.json").write_text(json.dumps(out, indent=2))
    print(f"saved {cfg.output_dir / 'bgsub_benchmark.json'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
