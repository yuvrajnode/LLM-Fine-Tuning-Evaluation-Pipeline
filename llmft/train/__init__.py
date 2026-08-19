"""Training stages. Imports are lazy - `import llmft.train` must not drag in torch."""

from __future__ import annotations

__all__ = ["run_sft", "run_dpo", "train_reward_model"]


def __getattr__(name: str):
    if name == "run_sft":
        from llmft.train.sft import run_sft

        return run_sft
    if name == "run_dpo":
        from llmft.train.dpo import run_dpo

        return run_dpo
    if name == "train_reward_model":
        from llmft.train.reward import train_reward_model

        return train_reward_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
