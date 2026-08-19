"""The multi-checkpoint evaluation harness.

This is the part that made iteration bearable. Before it existed, comparing
checkpoints meant loading each one by hand, generating into a scratch file and
eyeballing the output. Now `llmft eval` walks every checkpoint a run produced,
scores them on the same prompts with the same decoding settings, caches what it
has already computed, and writes one report the dashboard reads.

Two things do most of the work:

* The base model is loaded exactly once and adapters are swapped on top of it,
  instead of re-reading ~14GB of weights per checkpoint.
* Results are cached by checkpoint fingerprint, so adding a metric or a new
  checkpoint only evaluates what actually changed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmft.config import PipelineConfig
from llmft.data.loaders import SFTRecord, load_sft_records
from llmft.eval.metrics import score_batch
from llmft.eval.registry import Checkpoint, ResultCache, discover_checkpoints
from llmft.eval.report import build_report, write_report
from llmft.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CheckpointResult:
    checkpoint: Checkpoint
    metrics: dict[str, float]
    num_examples: int
    seconds: float
    samples: list[dict[str, str]] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.checkpoint.to_dict(),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "num_examples": self.num_examples,
            "seconds": round(self.seconds, 3),
            "samples": self.samples,
            "from_cache": self.from_cache,
        }


def _decoding_params(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "max_new_tokens": cfg.eval.max_new_tokens,
        "temperature": cfg.eval.temperature,
        "top_p": cfg.eval.top_p,
        "do_sample": cfg.eval.temperature > 0,
    }


def generate(
    model, tokenizer, prompts: Sequence[str], *, batch_size: int, decoding: dict[str, Any]
) -> list[str]:
    """Batched greedy/sampled generation, returning only the completion text."""
    import torch

    # Decoder-only models need left padding for batched generation, otherwise
    # short prompts get their continuation appended after the pad run.
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    params = dict(decoding)
    if not params.get("do_sample"):
        # HF warns (loudly, once per batch) if these are set while sampling is off.
        params.pop("temperature", None)
        params.pop("top_p", None)

    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            batch = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **params,
                )

            prompt_len = batch["input_ids"].shape[1]
            for row in generated:
                text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                outputs.append(text.strip())
    finally:
        tokenizer.padding_side = original_side

    return outputs


def evaluate_checkpoint(
    checkpoint: Checkpoint,
    records: Sequence[SFTRecord],
    cfg: PipelineConfig,
    *,
    model=None,
    tokenizer=None,
) -> CheckpointResult:
    """Score a single checkpoint. Pass `model`/`tokenizer` to reuse a loaded pair."""
    from llmft.train.model import build_tokenizer, load_for_inference

    started = time.time()
    base_model = cfg.eval.base_model or cfg.model.name_or_path

    if tokenizer is None:
        tokenizer = build_tokenizer(cfg.model)
    if model is None:
        adapter = None if checkpoint.is_base else checkpoint.path
        model = load_for_inference(base_model, adapter, cfg.model)

    prompts = [r.prompt for r in records]
    references = [r.response for r in records]
    predictions = generate(
        model,
        tokenizer,
        prompts,
        batch_size=cfg.eval.batch_size,
        decoding=_decoding_params(cfg),
    )

    metrics = score_batch(predictions, references, cfg.eval.tasks)
    samples = [
        {"prompt": p, "reference": r, "prediction": pred}
        for p, r, pred in list(zip(prompts, references, predictions, strict=True))[:3]
    ]

    return CheckpointResult(
        checkpoint=checkpoint,
        metrics=metrics,
        num_examples=len(records),
        seconds=time.time() - started,
        samples=samples,
    )


def run_evaluation(cfg: PipelineConfig, *, limit: int | None = None) -> dict[str, Any]:
    """Evaluate every checkpoint of a run and write the report. Returns the report."""
    from llmft.train.model import build_tokenizer, load_for_inference

    base_model = cfg.eval.base_model or cfg.model.name_or_path
    checkpoints = discover_checkpoints(
        cfg.eval.checkpoint_dir,
        base_model=base_model,
        include_base=cfg.eval.include_base_model,
    )
    if not checkpoints:
        raise RuntimeError(f"no checkpoints found under {cfg.eval.checkpoint_dir}")

    records, _ = load_sft_records(
        cfg.eval.dataset_path, cfg.data, limit=limit if limit is not None else cfg.eval.limit
    )
    if not records:
        raise RuntimeError(f"evaluation dataset {cfg.eval.dataset_path} produced no examples")

    cache = ResultCache(cfg.eval.report_dir, enabled=cfg.eval.cache_results)
    decoding = _decoding_params(cfg)
    tokenizer = build_tokenizer(cfg.model)

    results: list[CheckpointResult] = []
    evaluated = 0
    sweep_started = time.time()

    for index, ckpt in enumerate(checkpoints, start=1):
        cache_key = cache.key(ckpt, cfg.eval.dataset_path, cfg.eval.tasks, decoding)
        cached = cache.get(cache_key)
        if cached is not None:
            log.info("[%d/%d] %s: cached", index, len(checkpoints), ckpt.name)
            results.append(
                CheckpointResult(
                    checkpoint=ckpt,
                    metrics=cached["metrics"],
                    num_examples=cached["num_examples"],
                    seconds=cached.get("seconds", 0.0),
                    samples=cached.get("samples", []),
                    from_cache=True,
                )
            )
            continue

        log.info("[%d/%d] evaluating %s (step %d)", index, len(checkpoints), ckpt.name, ckpt.step)
        adapter = None if ckpt.is_base else ckpt.path
        model = load_for_inference(base_model, adapter, cfg.model)
        try:
            result = evaluate_checkpoint(ckpt, records, cfg, model=model, tokenizer=tokenizer)
        finally:
            _release(model)

        cache.put(cache_key, result.to_dict())
        results.append(result)
        evaluated += 1
        log.info(
            "    %s in %.1fs",
            ", ".join(f"{k}={v:.4f}" for k, v in result.metrics.items()),
            result.seconds,
        )

    report = build_report(
        cfg, [r.to_dict() for r in results], wall_seconds=time.time() - sweep_started
    )
    written = write_report(report, cfg.eval.report_dir, cfg.eval.dashboard_data)

    log.info(
        "sweep done: %d checkpoint(s), %d evaluated, %d served from cache -> %s",
        len(results),
        evaluated,
        len(results) - evaluated,
        ", ".join(str(p) for p in written),
    )
    return report


def _release(model) -> None:
    """Free a checkpoint's memory before loading the next one."""
    import gc

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def resolve_report_path(report_dir: str | Path) -> Path:
    return Path(report_dir) / "report.json"
