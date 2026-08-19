"""Logging setup. Rich if it's installed, plain stdlib otherwise."""

from __future__ import annotations

import logging
import os

try:
    from rich.console import Console
    from rich.logging import RichHandler

    console = Console()
    _HANDLER: logging.Handler = RichHandler(
        console=console, rich_tracebacks=True, show_path=False, markup=False
    )
    _FORMAT = "%(message)s"
except ImportError:  # pragma: no cover - only hit on a bare install
    console = None
    _HANDLER = logging.StreamHandler()
    _FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("LLMFT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", handlers=[_HANDLER])
    # transformers is extremely chatty at INFO during multi-checkpoint sweeps
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
