#!/usr/bin/env python
"""Rate-distortion benchmark of the temporal DCT compressor.

Sweeps the ``--quality`` knob of :mod:`experiments.temporal_compress` on a
``temporal_frames/`` sequence, and for every operating point measures

  * the on-disk compression ratio (original plys vs ``temporal.npz`` + manifest),
  * the render PSNR of the restored frames against renders of the originals
    (rendered from the 4 static training cameras on a frame subsample),

plus the per-attribute retained-coefficient budget K at the default quality.

Run::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.compression_benchmark \
        --frames-dir splats/thesis/full150/temporal-splatfacto/thesis/temporal_frames
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tyro

from experiments.eval_temporal import build_eval_cameras, load_gaussians, render_gaussians
from experiments.temporal_compress import compress, decompress


@dataclass
class Config:
    frames_dir: Path
    data: Path = Path("../output/dataset")
    qualities: str = "0.98,0.99,0.995,0.999,0.9999"
    default_quality: float = 0.999
    max_coeffs: int = 64
    frame_stride: int = 5
    """Render every Nth frame for the PSNR measurement."""
    eval_scale: float = 0.5
    output: Path = Path("experiments/outputs/compression/compression_benchmark.json")
    device: str = "cuda"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


@torch.no_grad()
def render_set(frames_dir: Path, frames: List[int], cams, device: str) -> Dict[int, List[torch.Tensor]]:
    out = {}
    for f in frames:
        g = load_gaussians(frames_dir / f"{f:05d}.ply", device)
        out[f] = [render_gaussians(g, c["c2w"], c["K"], c["width"], c["height"])[0].cpu() for c in cams]
    return out


def main(cfg: Config) -> None:
    cams, _ = build_eval_cameras(cfg.data, cfg.eval_scale)
    plys = sorted(cfg.frames_dir.glob("*.ply"))
    num_frames = len(plys)
    frames = list(range(0, num_frames, cfg.frame_stride))
    raw_bytes = dir_size(cfg.frames_dir)
    print(f"{num_frames} frames, raw {raw_bytes/1e6:.1f} MB; rendering {len(frames)} reference frames")
    ref = render_set(cfg.frames_dir, frames, cams, cfg.device)

    sweep = []
    per_attr_K: Dict[str, int] = {}
    num_gaussians = None
    for q in [float(x) for x in cfg.qualities.split(",")]:
        tmp = Path(tempfile.mkdtemp(prefix="tcomp_"))
        comp_dir, rest_dir = tmp / "comp", tmp / "rest"
        stats = compress(str(cfg.frames_dir), str(comp_dir), quality=q, max_coeffs=cfg.max_coeffs)
        comp_bytes = dir_size(comp_dir)
        decompress(str(comp_dir), str(rest_dir))
        rest = render_set(rest_dir, frames, cams, cfg.device)
        mses = []
        for f in frames:
            for a, b in zip(ref[f], rest[f]):
                mses.append(float(((a - b) ** 2).mean()))
        psnr = float(-10.0 * np.log10(max(np.mean(mses), 1e-12)))
        point = {"quality": q, "ratio": raw_bytes / comp_bytes, "render_psnr": psnr,
                 "compressed_mb": comp_bytes / 1e6}
        sweep.append(point)
        print(f"q={q}: ratio={point['ratio']:.1f}x  renderPSNR={psnr:.1f} dB "
              f"({comp_bytes/1e6:.2f} MB)")
        if abs(q - cfg.default_quality) < 1e-12:
            manifest = json.loads((comp_dir / "manifest.json").read_text())
            num_gaussians = manifest.get("num_gaussians")
            for name, v in manifest.get("per_property", {}).items():
                if v.get("mode") == "dct":
                    per_attr_K[name] = int(v["K"])
        shutil.rmtree(tmp)

    out = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "num_frames": num_frames,
        "num_gaussians": num_gaussians,
        "raw_mb": raw_bytes / 1e6,
        "sweep": sweep,
        "per_attribute_K": per_attr_K,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(out, indent=2))
    print(f"saved {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
