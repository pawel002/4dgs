#!/usr/bin/env python
"""Full-sequence (150x4) mask evaluation of the two segmentation operating points:
the production single threshold (tau=0.06, e=2) and the shadow-aware hysteresis
(tau_hi=0.1, tau_lo=0.01, chroma=0.04) — the numbers quoted in the thesis.

Run::

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m experiments.seg_full_eval
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from experiments.seg_variants_benchmark import Config as SegCfg
from experiments.seg_variants_benchmark import load_rgb, make_mask, mask_metrics


@dataclass
class Config:
    dataset: Path = Path("../output/dataset")
    gt: Path = Path("../output/black-bg-frames")
    cameras: str = "static1,static2,static3,static4"
    num_frames: int = 150
    gt_threshold: float = 0.02
    device: str = "cuda"
    output: Path = Path("experiments/outputs/bgsub/seg_full_eval.json")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def main(cfg: Config) -> None:
    seg_cfg = SegCfg()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    cams = cfg.cameras.split(",")
    plates = {c: load_rgb(cfg.dataset / c / "images" / "0000.png", device) for c in cams}
    arms = {
        "production": ("mean", 0.06, 0.0, 0.04),
        "hysteresis": ("hyst+shadow", 0.1, 0.01, 0.04),
    }
    out = {"config": {k: str(v) for k, v in vars(cfg).items()}, "arms": {}}
    for name, (variant, thr, lo, chroma) in arms.items():
        per_frame = []
        for t in range(1, cfg.num_frames + 1):
            ms = []
            for c in cams:
                img = load_rgb(cfg.dataset / c / "images" / f"{t:04d}.png", device)
                gt_img = load_rgb(cfg.gt / c / "images" / f"{t:04d}.png", device)
                gt_mask = gt_img.mean(-1) > cfg.gt_threshold
                mask = make_mask(variant, img, plates[c], thr, lo, seg_cfg, chroma)
                m = mask_metrics(mask, gt_mask)
                m["psnr"] = psnr(img * mask[..., None], gt_img)
                ms.append(m)
            per_frame.append({k: float(np.mean([m[k] for m in ms])) for k in ("iou", "precision", "recall", "psnr")})
            if t % 50 == 0:
                print(f"[{name}] frame {t}/{cfg.num_frames}")
        summary = {k: float(np.mean([f[k] for f in per_frame])) for k in ("iou", "precision", "recall", "psnr")}
        summary["iou_min"] = float(np.min([f["iou"] for f in per_frame]))
        out["arms"][name] = {"summary": summary, "per_frame": per_frame}
        print(f"[{name}] {json.dumps(summary, indent=2)}")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(out, indent=2))
    print(f"saved {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
