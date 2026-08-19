"""Model and tokenizer construction.

Heavy imports (torch/transformers/peft) live inside the functions so that the
config, data and reporting modules stay importable on a machine with no GPU
stack installed - the eval report tooling and the tests rely on that.
"""

from __future__ import annotations

from typing import Any

from llmft.config import LoraConfig as LoraCfg
from llmft.config import ModelConfig
from llmft.utils.logging import get_logger

log = get_logger(__name__)

_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


def resolve_dtype(name: str):
    import torch

    if name not in _DTYPES:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return getattr(torch, name)


def build_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.resolved_tokenizer(),
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )

    # Most base checkpoints ship without a pad token. Reusing EOS is the usual
    # fix; padding left keeps batched generation aligned for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("no pad_token on %s - falling back to eos_token", cfg.resolved_tokenizer())
    tokenizer.padding_side = "right"  # flipped to "left" by the generation path
    tokenizer.model_max_length = cfg.max_seq_length
    return tokenizer


def _quantization_config(cfg: ModelConfig):
    if not cfg.load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=resolve_dtype(cfg.bnb_compute_dtype),
    )


def load_base_model(cfg: ModelConfig, *, for_training: bool = True, **kwargs: Any):
    """Load the frozen base model, 4-bit quantised unless the config says otherwise."""
    import torch
    from transformers import AutoModelForCausalLM

    quant = _quantization_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name_or_path,
        quantization_config=quant,
        torch_dtype=resolve_dtype(cfg.bnb_compute_dtype),
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
        **kwargs,
    )

    if for_training:
        model.config.use_cache = False  # incompatible with gradient checkpointing
        if quant is not None:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing
            )
    else:
        model.config.use_cache = True

    return model


def attach_lora(model, cfg: LoraCfg):
    """Wrap a base model with LoRA adapters and log the trainable-parameter share."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
        target_modules=cfg.target_modules or None,
        modules_to_save=cfg.modules_to_save or None,
    )
    model = get_peft_model(model, peft_cfg)
    trainable, total = count_parameters(model)
    log.info(
        "LoRA r=%d alpha=%d -> %s trainable of %s (%.3f%%)",
        cfg.r,
        cfg.alpha,
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / max(total, 1),
    )
    return model


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_for_inference(base_model: str, adapter_path: str | None, cfg: ModelConfig):
    """Load a checkpoint for evaluation: base weights plus an optional adapter."""
    model_cfg = ModelConfig(**{**cfg.__dict__, "name_or_path": base_model})
    model = load_base_model(model_cfg, for_training=False)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        # Merging costs a little memory but removes the adapter indirection from
        # every forward pass; on a 15-checkpoint sweep it paid for itself.
        try:
            model = model.merge_and_unload()
        except (RuntimeError, ValueError) as exc:
            log.warning("could not merge adapter %s (%s) - evaluating unmerged", adapter_path, exc)

    model.eval()
    return model
