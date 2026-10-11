# Showcase verification

Verified locally on 11 October 2026 on macOS / Apple Silicon with Python 3.13.3.

| Check | Result |
| --- | --- |
| Python regression suite | 111 passed |
| Playwright, Chromium desktop and mobile | 8 passed |
| Ruff and Black | Passed |
| Dependency consistency (`pip check`) | Passed |
| Bundled report regeneration | Matches checked-in data, excluding timestamp |
| Sample dataset generation | Passed |
| SFT smoke training | Passed; finite training and validation loss |
| DPO smoke training from SFT | Passed with separate frozen reference adapter |
| Reward model smoke training | Passed with gradient accumulation |
| Evaluation sweep | Base, saved checkpoint, and final adapter evaluated |
| Repeated evaluation | 3 cache hits, 0 model evaluations |

Training checks used `sshleifer/tiny-gpt2` and the bundled sample data. They verify
pipeline integration, not model quality. CUDA/4-bit Mistral training, multi-GPU
execution, and long-running convergence were not verified. PPO remains unsupported.

The tested model API family is Transformers 4.49.0, PEFT 0.14.0, and TRL 0.15.2.
The dependency bounds in `pyproject.toml` keep those interfaces consistent. The DPO
reference adapter follows the [TRL 0.15.2 DPO documentation](https://huggingface.co/docs/trl/v0.15.2/dpo_trainer).

CI runs Python checks on 3.10 and 3.12, plus desktop/mobile Chromium tests. A manually
triggered workflow also runs SFT, evaluation, and DPO smoke checks. Remote CI has
not been executed as part of this local review.

## Reproduce

```sh
python -m pytest
ruff check llmft tests scripts
black --check llmft tests scripts
npm ci
npx playwright install chromium
npm test
llmft sft --config configs/smoke.yaml --smoke --set train.eval_steps=1 --set train.logging_steps=1
llmft eval --config configs/smoke.yaml --limit 2 --set eval.max_new_tokens=8 --set eval.dashboard_data=null
llmft dpo --config configs/smoke.yaml --smoke --sft-adapter checkpoints/smoke/final --set data.train_path=data/sample/preferences.jsonl --set data.eval_path=null --set train.output_dir=checkpoints/dpo-smoke --set rlhf.max_prompt_length=64
llmft reward --config configs/smoke.yaml --smoke --set data.train_path=data/sample/preferences.jsonl --set train.output_dir=checkpoints/reward-smoke --set train.gradient_accumulation_steps=2
```
