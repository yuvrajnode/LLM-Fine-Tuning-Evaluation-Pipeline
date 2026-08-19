from llmft.eval.metrics import METRICS, get_metric, score_batch
from llmft.eval.registry import Checkpoint, ResultCache, discover_checkpoints
from llmft.eval.report import build_report, summarise, to_markdown, write_report

__all__ = [
    "METRICS",
    "get_metric",
    "score_batch",
    "Checkpoint",
    "ResultCache",
    "discover_checkpoints",
    "build_report",
    "summarise",
    "to_markdown",
    "write_report",
    "run_evaluation",
]


def __getattr__(name: str):
    # harness pulls in torch, so keep it off the import path for reporting-only use.
    if name == "run_evaluation":
        from llmft.eval.harness import run_evaluation

        return run_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
