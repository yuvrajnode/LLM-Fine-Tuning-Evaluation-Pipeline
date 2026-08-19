.PHONY: install dev test lint fmt sft dpo eval dashboard clean

install:
	pip install -r requirements.txt && pip install -e .

dev:
	pip install -r requirements-dev.txt && pip install -e .

test:
	pytest

lint:
	ruff check llmft tests

fmt:
	black llmft tests && ruff check --fix llmft tests

sft:
	llmft sft --config configs/sft_lora.yaml

dpo:
	llmft dpo --config configs/dpo.yaml

eval:
	llmft eval --config configs/eval.yaml

dashboard:
	python -m http.server 8080 --directory dashboard

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
