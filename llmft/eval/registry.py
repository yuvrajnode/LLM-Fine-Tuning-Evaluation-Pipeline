"""Checkpoint discovery.

A training run leaves behind a directory of `checkpoint-<step>/` folders plus a
`final/`. This module turns that into an ordered list the harness can iterate,
enriched with whatever the manifest callback recorded during training.

It also owns the result cache. Re-running a sweep after adding one metric should
not re-generate from every checkpoint again, so each result is keyed by
(checkpoint fingerprint, dataset, tasks, decoding params) and skipped if present.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llmft.utils.io import read_jsonl
from llmft.utils.logging import get_logger

log = get_logger(__name__)

_STEP_RE = re.compile(r"checkpoint-(\d+)$")
_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json")


@dataclass
class Checkpoint:
    name: str
    path: str
    step: int
    epoch: float | None = None
    stage: str = "sft"
    train_loss: float | None = None
    eval_loss: float | None = None
    is_base: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Identify a checkpoint by its adapter bytes, not just its path.

        Two runs both write `checkpoint-200`; hashing the file size and mtime of
        the adapter keeps their cached results apart without reading the weights.
        """
        if self.is_base:
            return hashlib.sha1(f"base::{self.path}".encode()).hexdigest()[:16]

        parts = [self.path]
        for filename in _ADAPTER_FILES:
            f = Path(self.path) / filename
            if f.exists():
                stat = f.stat()
                parts.append(f"{filename}:{stat.st_size}:{int(stat.st_mtime)}")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_manifest(root: Path) -> dict[int, dict[str, Any]]:
    manifest_path = root / "checkpoints.jsonl"
    if not manifest_path.exists():
        return {}
    entries: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(manifest_path):
        step = row.get("step")
        if isinstance(step, int):
            entries[step] = row  # later rows win: a resumed run overwrites the old one
    return entries


def discover_checkpoints(
    checkpoint_dir: str | Path,
    *,
    base_model: str | None = None,
    include_base: bool = True,
) -> list[Checkpoint]:
    """Enumerate every checkpoint under `checkpoint_dir`, oldest step first."""
    root = Path(checkpoint_dir)
    if not root.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {root}")

    manifest = _read_manifest(root)
    found: list[Checkpoint] = []

    if include_base and base_model:
        # Step 0 is the untuned model. Without it, a 3-point gain looks like a
        # number rather than a comparison.
        found.append(
            Checkpoint(
                name="base",
                path=base_model,
                step=0,
                stage="base",
                is_base=True,
                meta={"note": "untuned reference model"},
            )
        )

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = _STEP_RE.search(entry.name)
        is_final = entry.name == "final"
        if not match and not is_final:
            continue
        if not any((entry / f).exists() for f in _ADAPTER_FILES):
            log.warning("skipping %s - no adapter weights inside", entry)
            continue

        step = int(match.group(1)) if match else _infer_final_step(manifest)
        row = manifest.get(step, {})
        found.append(
            Checkpoint(
                name=entry.name,
                path=str(entry),
                step=step,
                epoch=row.get("epoch"),
                stage=row.get("stage", "sft"),
                train_loss=row.get("train_loss"),
                eval_loss=row.get("eval_loss"),
                meta={"saved_at": row.get("saved_at")} if row else {},
            )
        )

    found.sort(key=lambda c: (c.step, c.name))
    log.info("discovered %d checkpoint(s) under %s", len(found), root)
    return found


def _infer_final_step(manifest: dict[int, dict[str, Any]]) -> int:
    return max(manifest) if manifest else 10**9  # sorts last when unknown


class ResultCache:
    """On-disk cache of evaluation results, one JSON file per (checkpoint, task set)."""

    def __init__(self, report_dir: str | Path, enabled: bool = True):
        self.root = Path(report_dir) / "cache"
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(checkpoint: Checkpoint, dataset_path: str, tasks: list[str], decoding: dict) -> str:
        payload = json.dumps(
            {
                "ckpt": checkpoint.fingerprint(),
                "dataset": dataset_path,
                "tasks": sorted(tasks),
                "decoding": {k: decoding[k] for k in sorted(decoding)},
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:20]

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            log.warning("dropping unreadable cache entry %s", path)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self.root / f"{key}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
