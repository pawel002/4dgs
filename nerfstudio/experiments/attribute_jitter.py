#!/usr/bin/env python
"""Measure frame-to-frame *attribute jitter* of a temporal_frames/ sequence.

Because the primitive set is conserved, every per-gaussian attribute is a time series.
Genuine motion lives in the positions; for a rigid subject the non-positional attributes
(opacity, scale, rotation) should evolve slowly — their frame-to-frame fluctuation under a
generous tracking budget is optimizer noise. This script quantifies that noise directly:
for each attribute group it records the mean absolute frame-to-frame change per frame
(mean over gaussians and channels, in raw parameter units), which is what the temporal
smoothness regularizer damps and what the DCT compressor pays for.

Run::

    .venv/bin/python experiments/attribute_jitter.py \
        --frames-dir splats/thesis/full150/temporal-splatfacto/thesis2/temporal_frames \
        --output experiments/outputs/jitter/full150.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from experiments.temporal_compress import read_ply

GROUPS = {
    "position": ["x", "y", "z"],
    "opacity": ["opacity"],
    "scale": ["scale_0", "scale_1", "scale_2"],
    "rotation": ["rot_0", "rot_1", "rot_2", "rot_3"],
}


@dataclass
class Config:
    frames_dir: Path
    output: Path = Path("experiments/outputs/jitter/run.json")
    max_frames: int = -1


def main(cfg: Config) -> None:
    plys = sorted(cfg.frames_dir.glob("*.ply"))
    if cfg.max_frames > 0:
        plys = plys[: cfg.max_frames]
    assert plys, f"no plys in {cfg.frames_dir}"

    prev = None
    per_frame = {g: [] for g in GROUPS}
    for ply in plys:
        props, data = read_ply(str(ply))
        col = {p: data[:, i] for i, p in enumerate(props)}
        cur = {g: np.stack([col[p] for p in props_g], axis=-1) for g, props_g in GROUPS.items()}
        if prev is not None and all(prev[g].shape == cur[g].shape for g in GROUPS):
            for g in GROUPS:
                per_frame[g].append(float(np.abs(cur[g] - prev[g]).mean()))
        prev = cur

    out = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "num_frames": len(plys),
        "per_frame_jitter": per_frame,
        "summary": {g: float(np.mean(v)) for g, v in per_frame.items()},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(out, indent=2))
    print(f"saved {cfg.output}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
