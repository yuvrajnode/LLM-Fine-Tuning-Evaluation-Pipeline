"""Reward modelling.

Only needed for the PPO path - the default DPO route optimises preferences
directly and never materialises a reward model. It is kept because the reward
model is still the most convenient way to *score* generations during evaluation,
independent of which algorithm produced them.

Loss is the standard Bradley-Terry pairwise objective:

    L = -log sigmoid( r(chosen) - r(rejected) )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from llmft.config import PipelineConfig
from llmft.data.loaders import load_preference_records
from llmft.train.model import attach_lora, build_tokenizer
from llmft.utils.logging import get_logger
from llmft.utils.seed import seed_everything

log = get_logger(__name__)


class PairwiseCollator:
    """Flatten (chosen, rejected) into one batch of 2N sequences.

    Both halves go through the model in a single forward pass, which keeps the
    two scores on the same graph and roughly halves wall-clock versus two passes.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: Sequence[dict[str, str]]) -> dict[str, Any]:
        texts = [f["prompt"] + f["chosen"] for f in features]
        texts += [f["prompt"] + f["rejected"] for f in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["num_pairs"] = len(features)
        return batch


def pairwise_loss(chosen_rewards, rejected_rewards, margin: float = 0.0):
    """Bradley-Terry loss plus the accuracy that actually tells you if it works."""
    import torch
    import torch.nn.functional as F

    diff = chosen_rewards - rejected_rewards - margin
    loss = -F.logsigmoid(diff).mean()
    accuracy = (diff > 0).float().mean()
    return loss, accuracy.detach()


def train_reward_model(cfg: PipelineConfig, *, smoke: bool = False) -> str:
    """Train a scalar reward head on preference pairs. Returns the output dir."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    seed_everything(cfg.train.seed)
    output_dir = Path(cfg.train.output_dir) / "reward-model"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg.model)
    pairs = load_preference_records(
        cfg.data.train_path, cfg.data, limit=32 if smoke else cfg.data.max_examples
    )
    if not pairs:
        raise RuntimeError(f"no usable preference pairs in {cfg.data.train_path}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=1,
        torch_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = attach_lora(model, _sequence_classification_lora(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loader = DataLoader(
        pairs,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=True,
        collate_fn=PairwiseCollator(tokenizer, cfg.model.max_seq_length),
    )
    epochs = 1 if smoke else int(max(1, cfg.train.epochs))
    total_steps = max(1, len(loader) * epochs)

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.train.learning_rate
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, int(total_steps * cfg.train.warmup_ratio), total_steps
    )

    model.train()
    step, running_acc = 0, 0.0
    for epoch in range(epochs):
        for batch in loader:
            num_pairs = batch.pop("num_pairs")
            batch = {k: v.to(device) for k, v in batch.items()}

            rewards = model(**batch).logits.squeeze(-1)
            loss, acc = pairwise_loss(rewards[:num_pairs], rewards[num_pairs:])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)

            step += 1
            running_acc += acc.item()
            if step % cfg.train.logging_steps == 0:
                log.info(
                    "reward step %d/%d | loss %.4f | pair acc %.3f",
                    step,
                    total_steps,
                    loss.item(),
                    running_acc / cfg.train.logging_steps,
                )
                running_acc = 0.0
        log.info("reward model: finished epoch %d", epoch + 1)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "reward_run.json").write_text(
        json.dumps({"pairs": len(pairs), "steps": step, "base_model": cfg.model.name_or_path}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    log.info("reward model saved -> %s", output_dir)
    return str(output_dir)


def _sequence_classification_lora(cfg: PipelineConfig):
    """Same LoRA settings as SFT but with the classification task type and the
    scalar head left trainable - freezing the head makes the loss go nowhere."""
    from dataclasses import replace

    return replace(cfg.lora, task_type="SEQ_CLS", modules_to_save=["score"])
