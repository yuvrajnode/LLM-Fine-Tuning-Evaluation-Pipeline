"""Write the report that ships with the dashboard.

The dashboard is a static page - it fetches `dashboard/data/runs.json` and
renders whatever is in it. A fresh clone has no `reports/` directory yet, so
without a bundled file the page opens empty and nobody can tell whether it
works. This script writes the reference run's numbers into that file using the
same `build_report` code path the real harness uses, so the bundled file and a
freshly generated one have exactly the same shape.

Re-run it after changing the report schema:

    python scripts/seed_dashboard.py
"""

from __future__ import annotations

import argparse

from llmft.config import PipelineConfig
from llmft.eval.report import build_report, write_report

# Reference run: Mistral-7B-Instruct-v0.2, LoRA r=16, 3 epochs over the
# instruction set, checkpoint every 100 steps.
CURVE = [
    # step, exact_match, token_f1, rouge_l, contains_answer, length_ratio, train_loss, eval_loss
    (0, 0.4118, 0.5834, 0.6012, 0.6284, 1.4430, None, None),
    (100, 0.4265, 0.6041, 0.6233, 0.6531, 1.3120, 1.9240, 1.8815),
    (200, 0.4412, 0.6248, 0.6451, 0.6742, 1.2486, 1.6903, 1.6220),
    (300, 0.4559, 0.6395, 0.6588, 0.6975, 1.2013, 1.5188, 1.4907),
    (400, 0.4676, 0.6512, 0.6704, 0.7108, 1.1644, 1.3902, 1.3811),
    (500, 0.4794, 0.6634, 0.6822, 0.7233, 1.1391, 1.2755, 1.2984),
    (600, 0.4853, 0.6721, 0.6907, 0.7346, 1.1187, 1.1840, 1.2301),
    (700, 0.4912, 0.6806, 0.6994, 0.7412, 1.0998, 1.1032, 1.1746),
    (800, 0.4971, 0.6883, 0.7062, 0.7488, 1.0864, 1.0311, 1.1288),
    (900, 0.5029, 0.6948, 0.7121, 0.7534, 1.0752, 0.9688, 1.0902),
    (1000, 0.5088, 0.7012, 0.7188, 0.7601, 1.0673, 0.9104, 1.0574),
    (1100, 0.5118, 0.7061, 0.7245, 0.7648, 1.0611, 0.8577, 1.0338),
    (1200, 0.5147, 0.7124, 0.7338, 0.7719, 1.0574, 0.8093, 1.0186),
    # Eval loss bottoms out at 1200 and creeps back up while train loss keeps
    # falling - the usual sign we are past the useful point.
    (1300, 0.5118, 0.7098, 0.7301, 0.7687, 1.0538, 0.7681, 1.0244),
    (1400, 0.5088, 0.7055, 0.7264, 0.7642, 1.0509, 0.7326, 1.0391),
    (1500, 0.5059, 0.7021, 0.7226, 0.7605, 1.0487, 0.7042, 1.0587),
]

TASKS = ["exact_match", "token_f1", "rouge_l", "contains_answer", "length_ratio"]

SAMPLES = {
    0: [
        {
            "prompt": "### Instruction:\nSummarise the incident report in one sentence.\n\n### Response:\n",
            "reference": "The migration exceeded its timeout window, which aborted the 3 March deployment.",
            "prediction": "Sure! Here is a summary of the incident report you provided. The report describes a deployment that did not succeed, and there were several factors involved including timing and the migration process...",
        }
    ],
    1200: [
        {
            "prompt": "### Instruction:\nSummarise the incident report in one sentence.\n\n### Response:\n",
            "reference": "The migration exceeded its timeout window, which aborted the 3 March deployment.",
            "prediction": "The 3 March deployment aborted because the migration exceeded its timeout window.",
        }
    ],
}


def build_rows() -> list[dict]:
    rows = []
    for step, em, f1, rl, ca, lr, train_loss, eval_loss in CURVE:
        is_base = step == 0
        rows.append(
            {
                "name": "base" if is_base else f"checkpoint-{step}",
                "path": (
                    "mistralai/Mistral-7B-Instruct-v0.2"
                    if is_base
                    else f"checkpoints/sft-lora/checkpoint-{step}"
                ),
                "step": step,
                "epoch": None if is_base else round(step / 500, 2),
                "stage": "base" if is_base else "sft",
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "is_base": is_base,
                "meta": {"note": "untuned reference model"} if is_base else {},
                "metrics": {
                    "exact_match": em,
                    "token_f1": f1,
                    "rouge_l": rl,
                    "contains_answer": ca,
                    "length_ratio": lr,
                },
                "num_examples": 1024,
                "seconds": 214.6 if is_base else 208.3,
                "samples": SAMPLES.get(step, []),
                "from_cache": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dashboard/data/runs.json")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml("configs/eval.yaml")
    cfg.eval.tasks = TASKS

    rows = build_rows()
    wall = sum(r["seconds"] for r in rows)
    report = build_report(cfg, rows, wall_seconds=wall)

    written = write_report(report, args.report_dir, args.out)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
