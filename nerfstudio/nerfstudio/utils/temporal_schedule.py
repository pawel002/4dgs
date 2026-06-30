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

"""Step <-> frame schedule for temporal (4D) gaussian splatting.

The training loop is a single, monotonically increasing sequence of steps, just
like vanilla splatfacto. We partition that sequence into one *initial*
reconstruction block for frame 0 (with densification) followed by one fixed-size
*tracking* block per subsequent frame::

    | <---- initial_iterations ----> | <- track -> | <- track -> | ...
    |            frame 0             |   frame 1   |   frame 2   | ...

Every component (datamanager, model, trainer) derives the active frame purely
from the global ``step`` using the helpers below, so they all agree without
having to share mutable state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


class EarlyStopper:
    """Plateau-based early stopping for per-frame gaussian fitting.

    Convergence metric: a quality score that should *increase* during fitting (PSNR by
    default). A frame is declared converged when either

    * the score reaches ``target`` (if an absolute target is given), or
    * the score has not improved by more than ``min_delta`` for ``patience`` consecutive
      updates (a plateau) — the standard early-stopping rule.

    This gives a natural, scene-agnostic "steps to convergence" for each frame and lets us
    compare how many steps velocity prediction saves to reach the *same* quality.

    Args:
        patience: number of non-improving updates tolerated before declaring a plateau.
        min_delta: minimum score increase (in metric units, e.g. dB for PSNR) that counts
            as an improvement.
        target: optional absolute score; reaching it stops immediately.
    """

    def __init__(self, patience: int = 30, min_delta: float = 0.03, target: Optional[float] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.target = target
        self.reset()

    def reset(self) -> None:
        self.best: float = -math.inf
        self.best_step: int = 0
        self._since_improved: int = 0
        self.converged: bool = False
        self.stopped_step: Optional[int] = None

    def update(self, value: float, step: int) -> bool:
        """Feed the latest metric ``value`` (observed at ``step``). Returns True once the
        frame is considered converged (and latches True on every later call)."""
        if self.converged:
            return True
        if self.target is not None and value >= self.target:
            self.best, self.best_step = value, step
            self.converged, self.stopped_step = True, step
            return True
        if value > self.best + self.min_delta:
            self.best, self.best_step = value, step
            self._since_improved = 0
        else:
            self._since_improved += 1
        if self._since_improved >= self.patience:
            self.converged, self.stopped_step = True, step
            return True
        return False


@dataclass(frozen=True)
class TemporalSchedule:
    """Maps global training steps to temporal frame indices.

    Args:
        num_frames: total number of temporal frames (>= 1).
        initial_iterations: steps spent reconstructing frame 0 (with densification).
        tracking_iterations: steps spent tracking each subsequent frame.
    """

    num_frames: int
    initial_iterations: int
    tracking_iterations: int

    @property
    def total_steps(self) -> int:
        """Number of steps required to process every frame exactly once."""
        return self.initial_iterations + max(0, self.num_frames - 1) * self.tracking_iterations

    def frame_of_step(self, step: int) -> int:
        """Return the frame index that step ``step`` belongs to (clamped to the last frame)."""
        if step < self.initial_iterations:
            return 0
        frame = 1 + (step - self.initial_iterations) // self.tracking_iterations
        return min(frame, self.num_frames - 1)

    def is_initial_frame(self, step: int) -> bool:
        """True while we are still doing the dense reconstruction of frame 0."""
        return step < self.initial_iterations

    def is_first_step_of_frame(self, step: int) -> bool:
        """True on the very first step of any frame (i.e. a frame transition)."""
        if step <= 0:
            return True
        return self.frame_of_step(step) != self.frame_of_step(step - 1)

    def is_last_step_of_frame(self, step: int) -> bool:
        """True on the last step of any frame (the moment its result is final)."""
        if step >= self.total_steps - 1:
            return True
        return self.frame_of_step(step) != self.frame_of_step(step + 1)

    def frame_local_step(self, step: int) -> int:
        """Step index *within* the current frame's block (0-based)."""
        if step < self.initial_iterations:
            return step
        return (step - self.initial_iterations) % self.tracking_iterations

    def frame_length(self, frame: int) -> int:
        """Number of steps allocated to ``frame``."""
        return self.initial_iterations if frame == 0 else self.tracking_iterations
