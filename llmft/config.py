"""Typed configuration objects backed by YAML.

Every stage of the pipeline (SFT, reward modelling, DPO, evaluation) reads one
YAML file. The dataclasses below are the single source of truth for what those
files may contain - an unknown key is an error rather than a silent no-op,
because a typo'd `learning_rate` that quietly does nothing costs a GPU-hour.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from llmft.utils.io import read_yaml

T = TypeVar("T")


def _build(cls: type[T], raw: dict[str, Any], path: str) -> T:
    """Recursively construct `cls` from a plain dict, rejecting unknown keys.

    `from __future__ import annotations` turns every annotation into a string, so
    the nested-section types are resolved through get_type_hints rather than read
    off `Field.type` directly.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{path}: unknown option(s) {sorted(unknown)}. Valid keys: {sorted(known)}"
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        hint = hints.get(f.name)
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = _build(hint, value, f"{path}.{f.name}")
        kwargs[f.name] = value
    return cls(**kwargs)


@dataclass
class ModelConfig:
    name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False
    load_in_4bit: bool = True
    bnb_compute_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True

    def resolved_tokenizer(self) -> str:
        return self.tokenizer_name_or_path or self.name_or_path


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Left empty so PEFT can auto-detect; override per architecture when the
    # auto-detection picks up the wrong projections (it does for some MoE models).
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    modules_to_save: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("lora.r must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("lora.dropout must be in [0, 1)")


@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"
    eval_path: str | None = "data/validation.jsonl"
    prompt_field: str = "instruction"
    context_field: str | None = "input"
    response_field: str = "output"
    template: str = "alpaca"
    max_examples: int | None = None
    shuffle_seed: int = 13
    # Preference data (DPO / reward model) uses a different shape.
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints/sft-lora"
    epochs: float = 3.0
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 20
    eval_steps: int | None = 100
    bf16: bool = True
    optim: str = "paged_adamw_8bit"
    seed: int = 13
    resume_from_checkpoint: str | None = None
    report_to: str = "none"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


@dataclass
class RLHFConfig:
    """Preference-optimisation settings.

    We landed on DPO as the default: it needs one fewer model in memory than PPO
    and, on this dataset, matched PPO's reward-model score within noise.
    """

    algorithm: str = "dpo"  # dpo | ipo | ppo
    beta: float = 0.1
    reference_free: bool = False
    max_prompt_length: int = 512
    reward_model_path: str | None = None
    # PPO-only knobs; ignored for DPO/IPO.
    ppo_epochs: int = 4
    kl_target: float = 6.0
    init_kl_coef: float = 0.05

    def __post_init__(self) -> None:
        allowed = {"dpo", "ipo", "ppo"}
        if self.algorithm not in allowed:
            raise ValueError(f"rlhf.algorithm must be one of {sorted(allowed)}")


@dataclass
class EvalConfig:
    checkpoint_dir: str = "checkpoints/sft-lora"
    base_model: str | None = None
    tasks: list[str] = field(default_factory=lambda: ["exact_match", "rouge_l"])
    dataset_path: str = "data/validation.jsonl"
    batch_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    limit: int | None = None
    report_dir: str = "reports"
    dashboard_data: str | None = "dashboard/data/runs.json"
    include_base_model: bool = True
    # Skip a checkpoint if a report for the same (checkpoint, dataset, tasks)
    # already exists. This is what makes re-running a sweep cheap.
    cache_results: bool = True


@dataclass
class PipelineConfig:
    run_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        raw = read_yaml(path)
        return _build(cls, raw, str(path))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
