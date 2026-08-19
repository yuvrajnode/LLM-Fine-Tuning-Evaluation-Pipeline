"""Prompt templates.

Keeping the templates in one place matters more than it looks: the eval harness
has to render prompts *identically* to training, otherwise every checkpoint
scores badly for reasons that have nothing to do with the weights. Anything that
renders a prompt goes through this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    with_context: str
    without_context: str
    response_prefix: str

    def render(self, instruction: str, context: str | None = None) -> str:
        instruction = (instruction or "").strip()
        context = (context or "").strip()
        if context:
            return self.with_context.format(instruction=instruction, context=context)
        return self.without_context.format(instruction=instruction)

    def render_full(self, instruction: str, response: str, context: str | None = None) -> str:
        return self.render(instruction, context) + (response or "").strip()


ALPACA = PromptTemplate(
    name="alpaca",
    with_context=(
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes "
        "the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{context}\n\n"
        "### Response:\n"
    ),
    without_context=(
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    ),
    response_prefix="### Response:\n",
)

CHATML = PromptTemplate(
    name="chatml",
    with_context=(
        "<|im_start|>user\n{instruction}\n\n{context}<|im_end|>\n<|im_start|>assistant\n"
    ),
    without_context="<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    response_prefix="<|im_start|>assistant\n",
)

# Mistral/Llama-2 style. No system turn - most of the small instruct models we
# benchmarked ignore it anyway when it is stuffed into [INST].
INST = PromptTemplate(
    name="inst",
    with_context="[INST] {instruction}\n\n{context} [/INST] ",
    without_context="[INST] {instruction} [/INST] ",
    response_prefix="[/INST] ",
)

PLAIN = PromptTemplate(
    name="plain",
    with_context="{instruction}\n\n{context}\n\n",
    without_context="{instruction}\n\n",
    response_prefix="",
)

TEMPLATES: dict[str, PromptTemplate] = {t.name: t for t in (ALPACA, CHATML, INST, PLAIN)}


def get_template(name: str) -> PromptTemplate:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise KeyError(f"unknown template {name!r}; available: {sorted(TEMPLATES)}") from None
