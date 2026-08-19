from __future__ import annotations

import json

import pytest

from llmft.config import DataConfig


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig(
        train_path="unused",
        eval_path=None,
        prompt_field="instruction",
        context_field="input",
        response_field="output",
        template="alpaca",
    )


@pytest.fixture
def jsonl(tmp_path):
    """Write rows to a .jsonl file and return the path."""

    def _write(name: str, rows):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return str(path)

    return _write
