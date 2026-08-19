"""Trainer callbacks.

The manifest callback is the glue between training and evaluation: every time
the Trainer writes a checkpoint, it appends a row to `checkpoints.jsonl`. The
eval harness reads that file instead of guessing at directory names, so a sweep
knows the step, epoch and loss behind each checkpoint without re-deriving them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from llmft.utils.logging import get_logger

log = get_logger(__name__)

try:  # transformers is optional for the reporting-only install
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover

    class TrainerCallback:  # type: ignore[no-redef]
        pass


class ThroughputCallback(TrainerCallback):
    """Log samples/sec and a rough ETA. `logging_steps` alone doesn't tell you
    whether the run will finish before you need the GPU back."""

    def __init__(self) -> None:
        self._start = 0.0
        self._last_step = 0
        self._last_time = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = self._last_time = time.time()
        self._last_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        steps = state.global_step - self._last_step
        elapsed = now - self._last_time
        if steps <= 0 or elapsed <= 0:
            return

        per_step = elapsed / steps
        samples_per_sec = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        ) / per_step
        remaining = max(state.max_steps - state.global_step, 0) * per_step

        log.info(
            "step %d/%d | %.2f samples/s | eta %s",
            state.global_step,
            state.max_steps,
            samples_per_sec,
            _fmt_duration(remaining),
        )
        self._last_step, self._last_time = state.global_step, now


class CheckpointManifestCallback(TrainerCallback):
    """Record each saved checkpoint so the eval harness can enumerate them."""

    def __init__(self, output_dir: str | Path, run_name: str, stage: str = "sft"):
        self.path = Path(output_dir) / "checkpoints.jsonl"
        self.run_name = run_name
        self.stage = stage
        self._latest_loss: float | None = None
        self._latest_eval_loss: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self._latest_loss = float(logs["loss"])
        if "eval_loss" in logs:
            self._latest_eval_loss = float(logs["eval_loss"])

    def on_save(self, args, state, control, **kwargs):
        entry = {
            "run_name": self.run_name,
            "stage": self.stage,
            "path": f"checkpoint-{state.global_step}",
            "step": state.global_step,
            "epoch": round(float(state.epoch or 0.0), 4),
            "train_loss": self._latest_loss,
            "eval_loss": self._latest_eval_loss,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
