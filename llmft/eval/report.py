"""Report assembly.

One JSON document per sweep, written to `reports/` and (optionally) copied to
`dashboard/data/runs.json`. The dashboard is a static page, so this file *is* the
API: keep the shape stable, and put anything the UI needs into it rather than
making the UI compute it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llmft.config import PipelineConfig
from llmft.eval.metrics import NON_DIRECTIONAL
from llmft.utils.io import write_json
from llmft.utils.logging import get_logger

log = get_logger(__name__)

REPORT_VERSION = 2


def _primary_metric(tasks: Sequence[str]) -> str | None:
    """First task that has a meaningful direction - what "best" is judged on."""
    for task in tasks:
        if task not in NON_DIRECTIONAL:
            return task
    return None


def summarise(results: Sequence[dict[str, Any]], tasks: Sequence[str]) -> dict[str, Any]:
    """Headline numbers: the best checkpoint and how far it moved from the base."""
    primary = _primary_metric(tasks)
    if primary is None or not results:
        return {"primary_metric": primary, "best": None, "baseline": None, "delta": None}

    scored = [r for r in results if primary in (r.get("metrics") or {})]
    if not scored:
        return {"primary_metric": primary, "best": None, "baseline": None, "delta": None}

    best = max(scored, key=lambda r: r["metrics"][primary])
    baseline = next((r for r in scored if r.get("is_base")), None)

    summary: dict[str, Any] = {
        "primary_metric": primary,
        "best": {
            "name": best["name"],
            "step": best["step"],
            "stage": best.get("stage"),
            "score": round(best["metrics"][primary], 6),
        },
        "baseline": None,
        "delta": None,
        "delta_pct": None,
    }

    if baseline is not None:
        base_score = baseline["metrics"][primary]
        summary["baseline"] = {"name": baseline["name"], "score": round(base_score, 6)}
        summary["delta"] = round(best["metrics"][primary] - base_score, 6)
        if base_score > 0:
            summary["delta_pct"] = round(
                100 * (best["metrics"][primary] - base_score) / base_score, 2
            )

    return summary


def _timing(results: Sequence[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    cached = [r for r in results if r.get("from_cache")]
    evaluated = [r for r in results if not r.get("from_cache")]
    mean_eval = sum(r.get("seconds", 0.0) for r in evaluated) / len(evaluated) if evaluated else 0.0
    # What the cache saved us this run, assuming a cached checkpoint would have
    # cost about what a freshly evaluated one did.
    saved = mean_eval * len(cached)
    return {
        "wall_seconds": round(wall_seconds, 2),
        "evaluated": len(evaluated),
        "from_cache": len(cached),
        "mean_seconds_per_checkpoint": round(mean_eval, 2),
        "estimated_seconds_saved": round(saved, 2),
    }


def build_report(
    cfg: PipelineConfig, results: Sequence[dict[str, Any]], *, wall_seconds: float = 0.0
) -> dict[str, Any]:
    tasks = list(cfg.eval.tasks)
    ordered = sorted(results, key=lambda r: (r.get("step", 0), r.get("name", "")))

    return {
        "report_version": REPORT_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": {
            "name": cfg.run_name,
            "base_model": cfg.eval.base_model or cfg.model.name_or_path,
            "checkpoint_dir": cfg.eval.checkpoint_dir,
            "dataset": cfg.eval.dataset_path,
            "num_examples": ordered[0]["num_examples"] if ordered else 0,
            "lora": {"r": cfg.lora.r, "alpha": cfg.lora.alpha, "dropout": cfg.lora.dropout},
            "decoding": {
                "max_new_tokens": cfg.eval.max_new_tokens,
                "temperature": cfg.eval.temperature,
                "top_p": cfg.eval.top_p,
            },
        },
        "tasks": tasks,
        "summary": summarise(ordered, tasks),
        "timing": _timing(ordered, wall_seconds),
        "checkpoints": list(ordered),
    }


def write_report(
    report: dict[str, Any], report_dir: str | Path, dashboard_path: str | Path | None = None
) -> list[Path]:
    """Write report.json, a timestamped copy, a markdown table, and the dashboard feed."""
    report_dir = Path(report_dir)
    stamp = report["generated_at"].replace(":", "").replace("-", "")
    written: list[Path] = []

    for target in (report_dir / "report.json", report_dir / f"report-{stamp}.json"):
        write_json(target, report)
        written.append(target)

    markdown = report_dir / "report.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(to_markdown(report), encoding="utf-8")
    written.append(markdown)

    if dashboard_path:
        write_json(dashboard_path, report)
        written.append(Path(dashboard_path))

    return written


def to_markdown(report: dict[str, Any]) -> str:
    """A table you can paste into a PR description."""
    tasks = report["tasks"]
    run = report["run"]
    summary = report.get("summary") or {}

    lines = [
        f"# Evaluation - {run['name']}",
        "",
        f"- Base model: `{run['base_model']}`",
        f"- Dataset: `{run['dataset']}` ({run['num_examples']} examples)",
        f"- LoRA: r={run['lora']['r']}, alpha={run['lora']['alpha']}",
        f"- Generated: {report['generated_at']}",
        "",
    ]

    if summary.get("best"):
        best = summary["best"]
        line = f"**Best checkpoint:** `{best['name']}` (step {best['step']}) - {summary['primary_metric']} = {best['score']:.4f}"
        if summary.get("delta_pct") is not None:
            line += f", {summary['delta_pct']:+.1f}% vs base"
        lines += [line, ""]

    header = "| checkpoint | step | " + " | ".join(tasks) + " |"
    divider = "|---" * (len(tasks) + 2) + "|"
    lines += [header, divider]

    for row in report["checkpoints"]:
        scores = " | ".join(f"{row['metrics'].get(t, float('nan')):.4f}" for t in tasks)
        lines.append(f"| {row['name']} | {row['step']} | {scores} |")

    timing = report.get("timing") or {}
    if timing:
        lines += [
            "",
            f"_{timing.get('evaluated', 0)} evaluated, {timing.get('from_cache', 0)} from cache, "
            f"{timing.get('wall_seconds', 0):.0f}s wall clock._",
        ]

    return "\n".join(lines) + "\n"
