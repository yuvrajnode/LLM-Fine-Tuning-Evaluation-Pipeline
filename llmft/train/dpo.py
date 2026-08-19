"""Preference optimisation (stage 2).

Runs on top of the stage-1 SFT adapter. DPO is the default; IPO is the same
trainer with a different loss, and PPO is available for comparison but needs a
trained reward model and a lot more memory.

Note on the reference model: with LoRA you do not need a second copy of the base
weights. Disabling the adapters gives you the reference policy for free, which is
what `ref_model=None` means to TRL when the policy is a PeftModel.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmft.config import PipelineConfig
from llmft.data.loaders import load_preference_records
from llmft.train.callbacks import CheckpointManifestCallback, ThroughputCallback
from llmft.train.model import attach_lora, build_tokenizer, load_base_model
from llmft.utils.logging import get_logger
from llmft.utils.seed import seed_everything

log = get_logger(__name__)


def _load_policy(cfg: PipelineConfig, sft_adapter: str | None):
    """Load the base model and either resume the SFT adapter or start fresh."""
    model = load_base_model(cfg.model, for_training=True)

    if sft_adapter and Path(sft_adapter).exists():
        from peft import PeftModel

        log.info("continuing from SFT adapter %s", sft_adapter)
        return PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)

    log.warning("no SFT adapter found - running DPO directly on the base model")
    return attach_lora(model, cfg.lora)


def run_dpo(cfg: PipelineConfig, *, sft_adapter: str | None = None, smoke: bool = False) -> str:
    """Run preference optimisation and return the output directory."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    if cfg.rlhf.algorithm == "ppo":
        raise NotImplementedError(
            "PPO is configured but not wired into this entrypoint. Use algorithm: dpo "
            "or algorithm: ipo, or call llmft.train.ppo directly once a reward model exists."
        )

    seed_everything(cfg.train.seed)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg.model)
    pairs = load_preference_records(
        cfg.data.train_path, cfg.data, limit=32 if smoke else cfg.data.max_examples
    )
    if not pairs:
        raise RuntimeError(f"no usable preference pairs in {cfg.data.train_path}")

    eval_pairs = None
    if cfg.data.eval_path and Path(cfg.data.eval_path).exists():
        eval_pairs = load_preference_records(
            cfg.data.eval_path, cfg.data, limit=32 if smoke else None
        )

    train_ds = Dataset.from_list(pairs)
    eval_ds = Dataset.from_list(eval_pairs) if eval_pairs else None

    policy = _load_policy(cfg, sft_adapter)

    args = DPOConfig(
        output_dir=str(output_dir),
        beta=cfg.rlhf.beta,
        loss_type="ipo" if cfg.rlhf.algorithm == "ipo" else "sigmoid",
        reference_free=cfg.rlhf.reference_free,
        max_length=cfg.model.max_seq_length,
        max_prompt_length=cfg.rlhf.max_prompt_length,
        num_train_epochs=0.05 if smoke else cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        logging_steps=cfg.train.logging_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        eval_strategy="steps" if (eval_ds and cfg.train.eval_steps) else "no",
        eval_steps=cfg.train.eval_steps,
        bf16=cfg.train.bf16,
        optim=cfg.train.optim,
        seed=cfg.train.seed,
        report_to=cfg.train.report_to,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,  # adapters get disabled to form the reference policy
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[
            ThroughputCallback(),
            CheckpointManifestCallback(output_dir, run_name=cfg.run_name, stage=cfg.rlhf.algorithm),
        ],
    )

    log.info(
        "starting %s: %d pairs, beta=%.3f, effective batch %d",
        cfg.rlhf.algorithm.upper(),
        len(pairs),
        cfg.rlhf.beta,
        cfg.train.effective_batch_size,
    )
    result = trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "stage": cfg.rlhf.algorithm,
                "base_model": cfg.model.name_or_path,
                "sft_adapter": sft_adapter,
                "pairs": len(pairs),
                "train_metrics": result.metrics,
                "config": cfg.to_dict(),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    log.info("%s complete -> %s", cfg.rlhf.algorithm.upper(), output_dir)
    return str(output_dir)
