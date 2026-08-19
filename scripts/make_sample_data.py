"""Generate the tiny sample dataset that ships with the repo.

This is *not* the training data - the real run used ~10.4k curated instruction
examples that we can't redistribute. These 40-odd rows exist so that
`llmft sft --config configs/sft_lora.yaml --smoke` runs end to end on a laptop
and so the tests have something concrete to chew on.

Usage:
    python scripts/make_sample_data.py --out data/sample
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from llmft.utils.io import write_jsonl

TASKS = [
    ("Summarise the paragraph in one sentence.", "summarisation"),
    ("Rewrite the sentence so it is easier to read.", "style"),
    ("Extract every date mentioned in the text as a JSON list.", "extraction"),
    ("Explain the error message and suggest a fix.", "code"),
    ("Classify the review as positive, negative, or neutral.", "classification"),
    ("Convert the description into a SQL query.", "code"),
    ("List the three main risks described in the passage.", "analysis"),
    ("Translate the sentence into formal English.", "style"),
]

CONTEXTS = [
    "The deployment failed at 02:14 UTC on 3 March 2026 after the migration timed out.",
    "Battery life is fine but the screen flickers whenever brightness drops below 20%.",
    "Quarterly revenue rose 12% while support costs grew 31% over the same period.",
    "TypeError: cannot unpack non-sequence NoneType raised in handlers/ingest.py line 88.",
    "The library ships with a default timeout of 30 seconds, configurable per client.",
    "Users in the EU region reported slower cold starts after the January rollout.",
]

RESPONSES = [
    "The migration exceeded its timeout window, which aborted the 3 March deployment.",
    "Screen flicker appears only under 20% brightness; battery performance is unaffected.",
    "Revenue grew 12%, but a 31% rise in support costs outpaced it.",
    "A function returned None where a tuple was expected - guard the return value before unpacking.",
    "The default 30-second timeout can be overridden on each client instance.",
    "EU cold starts regressed following the January rollout.",
]

WEAK_RESPONSES = [
    "It failed.",
    "There is a problem with the screen and maybe the battery too.",
    "Revenue and costs both went up.",
    "Something is None. Try fixing it.",
    "There is a timeout somewhere in the config.",
    "Performance changed in some regions.",
]


def build_rows(n: int, rng: random.Random) -> list[dict[str, str]]:
    rows = []
    for i in range(n):
        instruction, category = TASKS[i % len(TASKS)]
        idx = i % len(CONTEXTS)
        # The ticket reference keeps every row textually distinct, otherwise the
        # loader's de-duplication throws half the sample set away.
        rows.append(
            {
                "instruction": instruction,
                "input": f"[ticket OPS-{1000 + i}] {CONTEXTS[idx]}",
                "output": RESPONSES[idx],
                "category": category,
                "id": f"sample-{i:04d}",
            }
        )
    rng.shuffle(rows)
    return rows


def build_preference_rows(n: int, rng: random.Random) -> list[dict[str, str]]:
    rows = []
    for i in range(n):
        instruction, _ = TASKS[i % len(TASKS)]
        idx = i % len(CONTEXTS)
        rows.append(
            {
                "instruction": instruction,
                "input": f"[ticket OPS-{2000 + i}] {CONTEXTS[idx]}",
                "chosen": RESPONSES[idx],
                "rejected": WEAK_RESPONSES[idx],
            }
        )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sample", help="output directory")
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--preferences", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)

    written = {
        "train.jsonl": write_jsonl(out / "train.jsonl", build_rows(args.train, rng)),
        "validation.jsonl": write_jsonl(
            out / "validation.jsonl", build_rows(args.validation, rng)
        ),
        "preferences.jsonl": write_jsonl(
            out / "preferences.jsonl", build_preference_rows(args.preferences, rng)
        ),
    }
    for name, count in written.items():
        print(f"{out / name}: {count} rows")


if __name__ == "__main__":
    main()
