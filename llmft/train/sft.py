"""Supervised fine-tuning with LoRA adapters.

Stage 1 of the pipeline. Reads instruction data, renders it through the shared
prompt template, masks the prompt tokens out of the loss, and trains adapters on
top of a frozen (usually 4-bit) base model.

Masking the prompt is the part people skip. Training on the instruction tokens
as well is not catastrophic, but it wastes capacity teaching the model to
reproduce prompts it will always be given.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llmft.config import PipelineConfig
from llmft.data.loaders import SFTRecord, load_sft_records, split_records
from llmft.train.callbacks import CheckpointManifestCallback, ThroughputCallback
from llmft.train.model import attach_lora, build_tokenizer, count_parameters, load_base_model
from llmft.utils.logging import get_logger
from llmft.utils.seed import seed_everything

log = get_logger(__name__)

IGNORE_INDEX = -100


def _tokenize(record: SFTRecord, tokenizer, max_length: int) -> dict[str, list[int]]:
    """Tokenise one example, masking prompt tokens out of the labels."""
    prompt_ids = tokenizer(record.prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(
        record.text + (tokenizer.eos_token or ""),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    labels = list(full_ids)
    # Truncation can cut into the prompt on very long examples; clamp so we
    # never mask past the end of the sequence.
    mask_upto = min(len(prompt_ids), len(labels))
    for i in range(mask_upto):
        labels[i] = IGNORE_INDEX

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class CausalCollator:
    """Pad a batch to its longest member. Labels pad with -100, not the pad id."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        multiple = self.pad_to_multiple_of
        target = ((longest + multiple - 1) // multiple) * multiple

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = target - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def build_datasets(cfg: PipelineConfig, tokenizer, *, smoke: bool = False):
    from datasets import Dataset

    limit = 32 if smoke else None
    train_records, stats = load_sft_records(cfg.data.train_path, cfg.data, limit=limit)
    if not train_records:
        raise RuntimeError(
            f"no usable training examples in {cfg.data.train_path} ({stats.summary()})"
        )

    if cfg.data.eval_path and Path(cfg.data.eval_path).exists():
        eval_records, _ = load_sft_records(cfg.data.eval_path, cfg.data, limit=limit)
    else:
        log.info("no eval_path on disk - holding out 5%% of the training set instead")
        train_records, eval_records = split_records(train_records, seed=cfg.data.shuffle_seed)

    max_len = cfg.model.max_seq_length
    to_rows = lambda rs: [_tokenize(r, tokenizer, max_len) for r in rs]  # noqa: E731

    train_ds = Dataset.from_list(to_rows(train_records)).shuffle(seed=cfg.data.shuffle_seed)
    eval_ds = Dataset.from_list(to_rows(eval_records))
    log.info("train=%d examples, eval=%d examples", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def run_sft(cfg: PipelineConfig, *, smoke: bool = False) -> str:
    """Run stage 1 and return the output directory holding the checkpoints."""
    from transformers import Trainer, TrainingArguments

    seed_everything(cfg.train.seed)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg.model)
    train_ds, eval_ds = build_datasets(cfg, tokenizer, smoke=smoke)

    model = attach_lora(load_base_model(cfg.model, for_training=True), cfg.lora)
    trainable, total = count_parameters(model)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=0.02 if smoke else cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        logging_steps=cfg.train.logging_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        eval_strategy="steps" if cfg.train.eval_steps else "no",
        eval_steps=cfg.train.eval_steps,
        bf16=cfg.train.bf16,
        optim=cfg.train.optim,
        seed=cfg.train.seed,
        report_to=cfg.train.report_to,
        save_safetensors=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalCollator(tokenizer.pad_token_id),
        callbacks=[
            ThroughputCallback(),
            CheckpointManifestCallback(output_dir, run_name=cfg.run_name, stage="sft"),
        ],
    )

    log.info(
        "starting SFT: effective batch %d, %s trainable params",
        cfg.train.effective_batch_size,
        f"{trainable:,}",
    )
    result = trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "stage": "sft",
                "base_model": cfg.model.name_or_path,
                "trainable_params": trainable,
                "total_params": total,
                "train_metrics": result.metrics,
                "config": cfg.to_dict(),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    log.info("SFT complete -> %s", output_dir)
    return str(output_dir)
