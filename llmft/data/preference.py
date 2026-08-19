"""Turning ranked model samples into preference pairs.

The RLHF stage needs (prompt, chosen, rejected) triples. We generate several
candidates per prompt, score them (human labels, a reward model, or a heuristic
during smoke tests), and pair them up. Two details that mattered in practice:

* Pairing every candidate against every other one blows the dataset up
  quadratically and over-weights prompts that happened to get more samples, so
  `max_pairs_per_prompt` caps it.
* Pairs whose scores are nearly tied are noise. `min_margin` drops them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence


@dataclass
class Candidate:
    text: str
    score: float


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    margin: float
    meta: dict[str, str] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "margin": round(self.margin, 6),
            **self.meta,
        }


def build_preference_pairs(
    prompt: str,
    candidates: Sequence[Candidate],
    *,
    min_margin: float = 0.05,
    max_pairs_per_prompt: int = 4,
) -> list[PreferenceExample]:
    """Pair up scored candidates for a single prompt, best-separated pairs first."""
    usable = [c for c in candidates if (c.text or "").strip()]
    if len(usable) < 2:
        return []

    pairs: list[PreferenceExample] = []
    for a, b in combinations(usable, 2):
        winner, loser = (a, b) if a.score >= b.score else (b, a)
        margin = winner.score - loser.score
        if margin < min_margin or winner.text.strip() == loser.text.strip():
            continue
        pairs.append(
            PreferenceExample(
                prompt=prompt, chosen=winner.text, rejected=loser.text, margin=margin
            )
        )

    # Widest margins are the cleanest supervision, so keep those.
    pairs.sort(key=lambda p: p.margin, reverse=True)
    return pairs[:max_pairs_per_prompt]


def build_dataset(
    grouped: Iterable[tuple[str, Sequence[Candidate]]],
    *,
    min_margin: float = 0.05,
    max_pairs_per_prompt: int = 4,
) -> list[PreferenceExample]:
    """Flatten many prompts' candidates into one preference dataset."""
    out: list[PreferenceExample] = []
    for prompt, candidates in grouped:
        out.extend(
            build_preference_pairs(
                prompt,
                candidates,
                min_margin=min_margin,
                max_pairs_per_prompt=max_pairs_per_prompt,
            )
        )
    return out
