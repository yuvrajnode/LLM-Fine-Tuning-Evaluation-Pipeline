from __future__ import annotations

import pytest

from llmft.config import LoraConfig, PipelineConfig, RLHFConfig


def write_yaml(tmp_path, body: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


class TestLoading:
    def test_defaults_apply_to_missing_sections(self, tmp_path):
        cfg = PipelineConfig.from_yaml(write_yaml(tmp_path, "run_name: only-a-name\n"))
        assert cfg.run_name == "only-a-name"
        assert cfg.lora.r == 16
        assert cfg.train.epochs == 3.0

    def test_nested_sections_are_built(self, tmp_path):
        cfg = PipelineConfig.from_yaml(
            write_yaml(tmp_path, "model:\n  max_seq_length: 512\ntrain:\n  epochs: 1\n")
        )
        assert cfg.model.max_seq_length == 512
        assert cfg.train.epochs == 1

    def test_unknown_key_is_rejected(self, tmp_path):
        # The whole point of the typed layer: a typo must not be a silent no-op.
        path = write_yaml(tmp_path, "train:\n  learnign_rate: 0.001\n")
        with pytest.raises(ValueError, match="unknown option"):
            PipelineConfig.from_yaml(path)

    def test_error_names_the_valid_keys(self, tmp_path):
        path = write_yaml(tmp_path, "lora:\n  rank: 8\n")
        with pytest.raises(ValueError, match="dropout"):
            PipelineConfig.from_yaml(path)

    def test_non_mapping_yaml_is_rejected(self, tmp_path):
        path = write_yaml(tmp_path, "- one\n- two\n")
        with pytest.raises(ValueError, match="mapping"):
            PipelineConfig.from_yaml(path)


class TestValidation:
    def test_lora_rank_must_be_positive(self):
        with pytest.raises(ValueError, match="lora.r"):
            LoraConfig(r=0)

    def test_lora_dropout_range(self):
        with pytest.raises(ValueError, match="dropout"):
            LoraConfig(dropout=1.0)

    def test_rlhf_algorithm_is_checked(self):
        with pytest.raises(ValueError, match="algorithm"):
            RLHFConfig(algorithm="grpo")

    def test_supported_algorithms(self):
        for algo in ("dpo", "ipo", "ppo"):
            assert RLHFConfig(algorithm=algo).algorithm == algo


class TestDerived:
    def test_effective_batch_size(self):
        cfg = PipelineConfig()
        cfg.train.per_device_batch_size = 4
        cfg.train.gradient_accumulation_steps = 8
        assert cfg.train.effective_batch_size == 32

    def test_tokenizer_falls_back_to_the_model(self):
        cfg = PipelineConfig()
        assert cfg.model.resolved_tokenizer() == cfg.model.name_or_path
        cfg.model.tokenizer_name_or_path = "some/other-tokenizer"
        assert cfg.model.resolved_tokenizer() == "some/other-tokenizer"

    def test_round_trips_to_a_dict(self):
        payload = PipelineConfig().to_dict()
        assert payload["lora"]["r"] == 16
        assert payload["eval"]["tasks"]


class TestShippedConfigs:
    """Every config in configs/ must actually parse - they are documentation."""

    @pytest.mark.parametrize("name", ["sft_lora.yaml", "dpo.yaml", "eval.yaml", "smoke.yaml"])
    def test_parses(self, name):
        cfg = PipelineConfig.from_yaml(f"configs/{name}")
        assert cfg.run_name
        assert cfg.eval.tasks
