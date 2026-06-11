from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields


@dataclass
class ModelParams:
    """Scene and model loading parameters."""
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    depths: str = ""
    resolution: int = -1
    white_background: bool = False
    train_test_exp: bool = False
    data_device: str = "cuda"
    eval: bool = False
    sh_degree: int = 3

    def __post_init__(self):
        if self.source_path:
            self.source_path = os.path.abspath(self.source_path)


@dataclass
class PipelineParams:
    """Rendering pipeline parameters."""
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False


@dataclass
class OptimizationParams:
    """Training optimisation parameters."""
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    exposure_lr_init: float = 0.01
    exposure_lr_final: float = 0.001
    exposure_lr_delay_steps: int = 0
    exposure_lr_delay_mult: float = 0.0
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    depth_l1_weight_init: float = 1.0
    depth_l1_weight_final: float = 0.01
    random_background: bool = False
    optimizer_type: str = "default"


# ---------------------------------------------------------------------------
# Config save / load  (JSON, with fallback for legacy Namespace(...) format)
# ---------------------------------------------------------------------------

def save_config(model_params: ModelParams, path: str):
    """Persist *ModelParams* as JSON so render scripts can reload them."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(model_params), f, indent=2)


def load_config(path: str) -> ModelParams:
    """Load *ModelParams* from *path*.

    Tries JSON first, then falls back to the legacy ``Namespace(...)`` format
    written by the original codebase so that old checkpoints still work.
    """
    with open(path) as f:
        content = f.read()
    known = {fld.name for fld in fields(ModelParams)}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Legacy format: "Namespace(source_path='...', ...)"
        from argparse import Namespace
        ns = eval(content)  # noqa: S307 – kept for backward compat only
        data = vars(ns)
    return ModelParams(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Short-flag expansion (keeps backward compat with full_eval.py etc.)
# ---------------------------------------------------------------------------

_SHORT_FLAGS: dict[str, str] = {
    "-s": "--source-path",
    "-m": "--model-path",
    "-i": "--images",
    "-d": "--depths",
    "-r": "--resolution",
    "-w": "--white-background",
}


def expand_args(argv: list[str] | None = None) -> list[str]:
    """Replace legacy single-letter flags with their long equivalents.

    This lets callers keep using ``-s /data -m /output`` even though
    *tyro* only knows the long forms.
    """
    import sys
    if argv is None:
        argv = sys.argv[1:]
    return [_SHORT_FLAGS.get(tok, tok) for tok in argv]
