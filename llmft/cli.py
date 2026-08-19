"""Command line entrypoint.

    llmft sft    --config configs/sft_lora.yaml
    llmft dpo    --config configs/dpo.yaml
    llmft eval   --config configs/eval.yaml
    llmft report --report reports/report.json

Torch and friends are imported inside the commands, so `llmft --help` and
`llmft report` stay fast and work without a GPU stack installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from llmft import __version__
from llmft.config import PipelineConfig
from llmft.utils.logging import get_logger

log = get_logger("llmft")

app = typer.Typer(
    add_completion=False,
    help="LoRA/RLHF fine-tuning and multi-checkpoint evaluation for open LLMs.",
    no_args_is_help=True,
)

ConfigOption = typer.Option(..., "--config", "-c", help="Path to a pipeline YAML config.")


def _load(config: Path, overrides: list[str] | None = None) -> PipelineConfig:
    cfg = PipelineConfig.from_yaml(config)
    for override in overrides or []:
        _apply_override(cfg, override)
    return cfg


def _apply_override(cfg: PipelineConfig, override: str) -> None:
    """Apply a `section.key=value` override so you don't need a new YAML file
    just to try a different learning rate."""
    if "=" not in override:
        raise typer.BadParameter(f"expected section.key=value, got {override!r}")

    dotted, raw = override.split("=", 1)
    target = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise typer.BadParameter(f"unknown config section {part!r} in {override!r}")
        target = getattr(target, part)

    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise typer.BadParameter(f"unknown config key {leaf!r} in {override!r}")

    current = getattr(target, leaf)
    setattr(target, leaf, _coerce(raw, current))
    log.info("override %s = %s", dotted, getattr(target, leaf))


def _coerce(raw: str, current):
    """Match the type of the value already in the config; fall back to JSON."""
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(current, str) or current is None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(f"llmft {__version__}")
        raise typer.Exit()


@app.command()
def sft(
    config: Path = ConfigOption,
    smoke: bool = typer.Option(False, "--smoke", help="Tiny run to prove the wiring works."),
    set_: list[str] = typer.Option(None, "--set", help="Override a config key: train.epochs=1"),
) -> None:
    """Stage 1: supervised fine-tuning with LoRA adapters."""
    from llmft.train.sft import run_sft

    out = run_sft(_load(config, set_), smoke=smoke)
    typer.echo(f"checkpoints -> {out}")


@app.command()
def dpo(
    config: Path = ConfigOption,
    sft_adapter: str = typer.Option(
        None, "--sft-adapter", help="Adapter from stage 1 to continue from."
    ),
    smoke: bool = typer.Option(False, "--smoke"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Stage 2: preference optimisation (DPO/IPO) on top of the SFT adapter."""
    from llmft.train.dpo import run_dpo

    out = run_dpo(_load(config, set_), sft_adapter=sft_adapter, smoke=smoke)
    typer.echo(f"checkpoints -> {out}")


@app.command("reward")
def reward(
    config: Path = ConfigOption,
    smoke: bool = typer.Option(False, "--smoke"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Train a scalar reward model on preference pairs (needed only for PPO)."""
    from llmft.train.reward import train_reward_model

    out = train_reward_model(_load(config, set_), smoke=smoke)
    typer.echo(f"reward model -> {out}")


@app.command("eval")
def evaluate(
    config: Path = ConfigOption,
    limit: int = typer.Option(None, "--limit", help="Only score the first N examples."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached results."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Benchmark every checkpoint a run produced and write the report."""
    from llmft.eval.harness import run_evaluation

    cfg = _load(config, set_)
    if no_cache:
        cfg.eval.cache_results = False

    report = run_evaluation(cfg, limit=limit)
    summary = report.get("summary") or {}
    if summary.get("best"):
        best = summary["best"]
        delta = summary.get("delta_pct")
        suffix = f" ({delta:+.1f}% vs base)" if delta is not None else ""
        typer.echo(
            f"best: {best['name']} - {summary['primary_metric']}={best['score']:.4f}{suffix}"
        )


@app.command()
def checkpoints(
    config: Path = ConfigOption,
) -> None:
    """List the checkpoints the harness would evaluate, without running anything."""
    from llmft.eval.registry import discover_checkpoints

    cfg = _load(config)
    found = discover_checkpoints(
        cfg.eval.checkpoint_dir,
        base_model=cfg.eval.base_model or cfg.model.name_or_path,
        include_base=cfg.eval.include_base_model,
    )
    for ckpt in found:
        loss = f"{ckpt.eval_loss:.4f}" if ckpt.eval_loss is not None else "-"
        typer.echo(f"{ckpt.step:>7}  {ckpt.name:<20} {ckpt.stage:<6} eval_loss={loss}")


@app.command()
def report(
    path: Path = typer.Option("reports/report.json", "--report", "-r"),
    markdown: bool = typer.Option(False, "--markdown", help="Print the markdown table."),
) -> None:
    """Print a previously generated report without re-running the sweep."""
    from llmft.eval.report import to_markdown

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if markdown:
        typer.echo(to_markdown(data))
        return

    tasks = data["tasks"]
    typer.echo(f"{'checkpoint':<22}{'step':>7}  " + "  ".join(f"{t:>14}" for t in tasks))
    for row in data["checkpoints"]:
        scores = "  ".join(f"{row['metrics'].get(t, float('nan')):>14.4f}" for t in tasks)
        typer.echo(f"{row['name']:<22}{row['step']:>7}  {scores}")


if __name__ == "__main__":
    app()
