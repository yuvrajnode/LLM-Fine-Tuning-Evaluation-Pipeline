"""Dataset loading and validation.

The loaders deliberately do the boring work up front - dropping empty rows,
de-duplicating, reporting how many examples were thrown away - because a silent
data problem shows up three hours later as a flat loss curve.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Sequence

from llmft.config import DataConfig
from llmft.data.formatting import get_template
from llmft.utils.io import read_jsonl
from llmft.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SFTRecord:
    """One supervised example, already rendered into prompt/response strings."""

    prompt: str
    response: str
    text: str
    meta: dict[str, Any]

    def fingerprint(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


@dataclass
class LoadStats:
    total: int = 0
    kept: int = 0
    dropped_missing: int = 0
    dropped_empty: int = 0
    dropped_duplicate: int = 0

    def summary(self) -> str:
        return (
            f"{self.kept}/{self.total} kept "
            f"(missing_field={self.dropped_missing}, empty={self.dropped_empty}, "
            f"duplicate={self.dropped_duplicate})"
        )


def load_sft_records(
    path: str,
    cfg: DataConfig,
    *,
    deduplicate: bool = True,
    limit: int | None = None,
) -> tuple[list[SFTRecord], LoadStats]:
    """Read a .jsonl instruction dataset into rendered SFT records."""
    template = get_template(cfg.template)
    stats = LoadStats()
    seen: set[str] = set()
    records: list[SFTRecord] = []

    for row in read_jsonl(path):
        stats.total += 1

        if cfg.prompt_field not in row or cfg.response_field not in row:
            stats.dropped_missing += 1
            continue

        instruction = str(row.get(cfg.prompt_field) or "").strip()
        response = str(row.get(cfg.response_field) or "").strip()
        context = None
        if cfg.context_field:
            context = str(row.get(cfg.context_field) or "").strip() or None

        if not instruction or not response:
            stats.dropped_empty += 1
            continue

        record = SFTRecord(
            prompt=template.render(instruction, context),
            response=response,
            text=template.render_full(instruction, response, context),
            meta={k: v for k, v in row.items() if k not in {cfg.prompt_field, cfg.response_field}},
        )

        if deduplicate:
            fp = record.fingerprint()
            if fp in seen:
                stats.dropped_duplicate += 1
                continue
            seen.add(fp)

        records.append(record)
        stats.kept += 1

        cap = limit if limit is not None else cfg.max_examples
        if cap is not None and len(records) >= cap:
            break

    log.info("loaded %s: %s", path, stats.summary())
    if stats.total and stats.kept / stats.total < 0.9:
        log.warning(
            "dropped %.1f%% of %s - check the field names in your data config",
            100 * (1 - stats.kept / stats.total),
            path,
        )
    return records, stats


def load_preference_records(
    path: str, cfg: DataConfig, *, limit: int | None = None
) -> list[dict[str, str]]:
    """Read preference triples (prompt, chosen, rejected) for DPO / reward modelling."""
    template = get_template(cfg.template)
    out: list[dict[str, str]] = []
    skipped = 0

    for row in read_jsonl(path):
        instruction = str(row.get(cfg.prompt_field) or "").strip()
        chosen = str(row.get(cfg.chosen_field) or "").strip()
        rejected = str(row.get(cfg.rejected_field) or "").strip()
        context = str(row.get(cfg.context_field) or "").strip() if cfg.context_field else ""

        # A pair where both sides are identical carries no gradient signal for
        # DPO, and TRL will happily train on it, so drop those here.
        if not instruction or not chosen or not rejected or chosen == rejected:
            skipped += 1
            continue

        out.append(
            {
                "prompt": template.render(instruction, context or None),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        cap = limit if limit is not None else cfg.max_examples
        if cap is not None and len(out) >= cap:
            break

    log.info("loaded %d preference pairs from %s (%d skipped)", len(out), path, skipped)
    return out


def split_records(
    records: Sequence[SFTRecord], eval_fraction: float = 0.05, seed: int = 13
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Hold out a validation slice when the config doesn't point at one."""
    if not 0.0 < eval_fraction < 0.5:
        raise ValueError("eval_fraction must be in (0, 0.5)")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[cut:], shuffled[:cut]
