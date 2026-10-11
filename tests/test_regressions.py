"""Regression coverage for errors found during the showcase review."""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from llmft.cli import _load, app
from llmft.config import DataConfig, EvalConfig, PipelineConfig
from llmft.data.loaders import load_sft_records, split_records
from llmft.eval.registry import Checkpoint, ResultCache, discover_checkpoints
from llmft.train.callbacks import CheckpointManifestCallback
from llmft.train.sft import _tokenize
from llmft.utils.io import read_jsonl


@pytest.mark.parametrize("body", ["false", "[]", "model: null", "train: 5"])
def test_invalid_config_sections_fail_early(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match="mapping"):
        PipelineConfig.from_yaml(path)


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"limit": 0}, {"top_p": 2}, {"tasks": []}])
def test_invalid_eval_settings_fail_early(options):
    with pytest.raises(ValueError):
        EvalConfig(**options)


def test_override_revalidates_dataclass(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}")
    with pytest.raises(ValueError, match="lora.r"):
        _load(path, ["lora.r=0"])


def test_invalid_boolean_override_does_not_silently_disable_feature(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}")
    with pytest.raises(Exception, match="boolean"):
        _load(path, ["model.load_in_4bit=tru"])


def test_cli_version_without_subcommand():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llmft" in result.stdout


def test_jsonl_rejects_non_objects_with_line_number(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("{}\n[]\n")
    with pytest.raises(ValueError, match=":2: expected a JSON object"):
        list(read_jsonl(path))


def test_zero_limit_is_not_one_example(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"instruction":"q","output":"a"}\n')
    with pytest.raises(ValueError, match="positive"):
        load_sft_records(str(path), DataConfig(), limit=0)


def test_split_requires_enough_examples():
    with pytest.raises(ValueError, match="at least two"):
        split_records([])


def test_cache_tracks_dataset_contents_and_evaluation_context(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("first")
    checkpoint = Checkpoint("base", "model", 0, is_base=True)

    def key(limit):
        return ResultCache.key(checkpoint, str(path), ["exact_match"], {}, context={"limit": limit})

    original = key(8)
    assert original != key(16)
    path.write_text("other")
    assert original != key(8)


def test_config_without_weights_is_not_a_checkpoint(tmp_path):
    folder = tmp_path / "checkpoint-10"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}")
    assert discover_checkpoints(tmp_path, include_base=False) == []


def test_final_step_is_recorded_without_periodic_save(tmp_path):
    callback = CheckpointManifestCallback(tmp_path, "smoke")
    callback.on_train_end(None, SimpleNamespace(global_step=1, epoch=0.1), None)
    final = tmp_path / "final"
    final.mkdir()
    (final / "adapter_model.safetensors").write_bytes(b"weights")
    assert discover_checkpoints(tmp_path, include_base=False)[0].step == 1


def test_tokenization_preserves_response_boundary():
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            return {"input_ids": {"prompt": [1, 2], "answer": [3, 4]}[text].copy()}

    record = SimpleNamespace(prompt="prompt", response="answer")
    row = _tokenize(record, Tokenizer(), 5)
    assert row["input_ids"] == [1, 2, 3, 4, 99]
    assert row["labels"] == [-100, -100, 3, 4, 99]


def test_cached_sweep_never_loads_tokenizer_or_model(tmp_path, monkeypatch):
    from llmft.eval.harness import run_evaluation
    from llmft.train import model

    cfg = PipelineConfig()
    cfg.eval.checkpoint_dir = str(tmp_path)
    cfg.eval.dataset_path = str(tmp_path / "data.jsonl")
    cfg.eval.report_dir = str(tmp_path / "reports")
    cfg.eval.dashboard_data = None
    (tmp_path / "data.jsonl").write_text(json.dumps({"instruction": "q", "output": "a"}) + "\n")

    def unexpected(*args, **kwargs):
        pytest.fail("cached sweep tried to load model assets")

    monkeypatch.setattr(model, "build_tokenizer", unexpected)
    monkeypatch.setattr(model, "load_for_inference", unexpected)
    monkeypatch.setattr(
        ResultCache, "get", lambda *args: {"metrics": {"exact_match": 1}, "num_examples": 1}
    )
    result = run_evaluation(cfg)
    assert result["timing"]["from_cache"] == 1


@pytest.mark.parametrize(
    "body",
    [
        'model:\n  load_in_4bit: "false"',
        "eval:\n  tasks: exact_match",
        "train:\n  per_device_batch_size: true",
    ],
)
def test_wrong_config_types_fail_before_training(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match="invalid type"):
        PipelineConfig.from_yaml(path)
