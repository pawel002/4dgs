#!/usr/bin/env python
"""Build the merged *background* dataset for the static room reconstruction.

The static background is reconstructed from the orbiting ``dynamic1`` camera. The four
fixed cameras additionally each contribute one photograph of the *empty* room, the
frame-0 background plate that the foreground segmentation consumes. Those plates are
genuine, calibrated views of the exact scene the background reconstruction targets
(the room without the subject), taken from viewpoints (the room corners) that the
orbit undersamples, so this script adds them to the background training set.

It writes ``<dataset>/background/transforms.json`` whose frames are the ``dynamic1``
orbit frames plus the four ``static*/images/0000.png`` plates (poses taken from each
static camera's transforms; all cameras share identical intrinsics). File paths are
relative (``../dynamic1/...``, ``../static1/...``) so no images are copied. The plate
frames are placed at indices that fall into nerfstudio's *train* split (fraction 0.9),
so all four plates train and the eval split stays orbit-only.

Run (CPU)::

    .venv/bin/python -m experiments.make_background_dataset
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Config:
    dataset: Path = Path("../output/dataset")
    orbit: str = "dynamic1"
    camera_prefix: str = "static"
    plate_name: str = "0000.png"
    out_name: str = "background"
    train_split_fraction: float = 0.9
    """Must match the nerfstudio-data dataparser setting used for training (used to
    place the plate frames at train-split indices)."""


def main(cfg: Config) -> None:
    orbit_meta = json.loads((cfg.dataset / cfg.orbit / "transforms.json").read_text())
    frames = []
    for f in orbit_meta["frames"]:
        nf = dict(f)
        nf["file_path"] = f"../{cfg.orbit}/{f['file_path']}"
        frames.append(nf)

    plates = []
    cam_dirs = sorted(d for d in cfg.dataset.iterdir() if d.is_dir() and d.name.startswith(cfg.camera_prefix))
    for cam in cam_dirs:
        meta = json.loads((cam / "transforms.json").read_text())
        for k in ("fl_x", "fl_y", "cx", "cy", "w", "h"):
            assert meta[k] == orbit_meta[k], f"{cam.name} intrinsic {k} differs from the orbit camera"
        plate = next(f for f in meta["frames"] if Path(f["file_path"]).name == cfg.plate_name)
        plates.append({"file_path": f"../{cam.name}/{plate['file_path']}",
                       "transform_matrix": plate["transform_matrix"]})
    assert plates, "no static plates found"

    # Append the plates, then swap any plate that would land in the eval split with the
    # nearest earlier train-split orbit frame, so every plate ends up training.
    frames += plates
    n = len(frames)
    n_train = math.ceil(n * cfg.train_split_fraction)
    i_train = set(np.linspace(0, n - 1, n_train, dtype=int).tolist())
    plate_lo = n - len(plates)
    for i in range(plate_lo, n):
        if i in i_train:
            continue
        j = next(j for j in range(plate_lo - 1, -1, -1) if j in i_train)
        frames[i], frames[j] = frames[j], frames[i]
        plate_lo = min(plate_lo, j)  # conservative: keep looking left for further swaps

    out_dir = cfg.dataset / cfg.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta = {k: v for k, v in orbit_meta.items() if k != "frames"}
    out_meta["frames"] = frames
    (out_dir / "transforms.json").write_text(json.dumps(out_meta, indent=2))
    n_plate_train = sum(1 for i, f in enumerate(frames) if "static" in f["file_path"] and i in i_train)
    print(f"wrote {out_dir/'transforms.json'}: {len(frames)} frames "
          f"({len(frames) - len(plates)} orbit + {len(plates)} plates, {n_plate_train}/{len(plates)} plates in train split)")


if __name__ == "__main__":
    main(tyro.cli(Config))
