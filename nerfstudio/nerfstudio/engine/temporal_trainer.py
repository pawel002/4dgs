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

"""Trainer for temporal (4D) gaussian splatting.

A thin wrapper around the standard :class:`Trainer` that:

* derives ``max_num_iterations`` from the temporal schedule
  (``initial_iterations + (num_frames - 1) * tracking_iterations``) so the run
  ends exactly when the last frame has been processed — the user does not have to
  compute the step count by hand;
* reconciles the model and datamanager onto one authoritative
  :class:`TemporalSchedule` (derived from the *model* config) so all three
  components agree even if only one was overridden on the CLI;
* tells the model where to write its per-frame ``.ply`` files
  (``<output>/temporal_frames/``).

The actual training loop, viewer, and nerfstudio ``.ckpt`` checkpointing are all
inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

from nerfstudio.data.dataparsers.temporal_dataparser import TemporalDataParser
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.temporal_schedule import TemporalSchedule


@dataclass
class TemporalTrainerConfig(TrainerConfig):
    """Configuration for the temporal trainer."""

    _target: Type = field(default_factory=lambda: TemporalTrainer)


class TemporalTrainer(Trainer):
    """Trainer that runs the per-frame temporal gaussian-splatting schedule."""

    def _model_schedule_params(self) -> tuple[int, int]:
        """Read (initial_iterations, tracking_iterations) from the model config."""
        model_config = self.config.pipeline.model
        return int(model_config.initial_iterations), int(model_config.tracking_iterations)

    def _resolve_data_path(self) -> Path:
        dataparser = self.config.pipeline.datamanager.dataparser
        return Path(self.config.data) if self.config.data is not None else Path(dataparser.data)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        # --- size the run before the pipeline / writer / loop read max_num_iterations ---
        initial_iterations, tracking_iterations = self._model_schedule_params()
        camera_prefix = getattr(self.config.pipeline.datamanager.dataparser, "camera_prefix", "static")
        max_frames = getattr(self.config.pipeline.datamanager.dataparser, "max_frames", -1)
        bg_subtract = getattr(self.config.pipeline.datamanager.dataparser, "bg_subtract", True)
        num_frames = TemporalDataParser.count_frames(
            self._resolve_data_path(), camera_prefix=camera_prefix, max_frames=max_frames, bg_subtract=bg_subtract
        )
        if num_frames <= 0:
            CONSOLE.print("[bold yellow][temporal] Could not count frames up front; falling back to config value.")
            num_frames = max(1, num_frames)

        schedule = TemporalSchedule(
            num_frames=num_frames,
            initial_iterations=initial_iterations,
            tracking_iterations=tracking_iterations,
        )
        self.config.max_num_iterations = schedule.total_steps
        CONSOLE.log(
            f"[temporal] {num_frames} frames -> max_num_iterations={schedule.total_steps} "
            f"(initial={initial_iterations}, tracking={tracking_iterations})."
        )

        super().setup(test_mode=test_mode)

        # --- reconcile model + datamanager onto the authoritative schedule ---
        model = self.pipeline.model
        datamanager = self.pipeline.datamanager
        if hasattr(model, "set_schedule"):
            model.set_schedule(schedule)
        if hasattr(datamanager, "set_schedule"):
            datamanager.set_schedule(schedule)

        # --- tell the model where to dump per-frame plys ---
        if hasattr(model, "temporal_ply_dir"):
            model.temporal_ply_dir = self.base_dir / "temporal_frames"
            CONSOLE.log(f"[temporal] per-frame plys -> {model.temporal_ply_dir}")
