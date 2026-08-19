"""Tests for checkpoint discovery, the result cache and report assembly.

These are the parts of the eval path that run without torch, and they are also
the parts most likely to break something silently: a checkpoint quietly skipped
or a stale cache hit looks like a training regression.
"""

from __future__ import annotations

import json

import pytest

from llmft.config import PipelineConfig
from llmft.eval.registry import ResultCache, discover_checkpoints
from llmft.eval.report import build_report, summarise, to_markdown, write_report


def make_checkpoint(root, step: int, *, adapter: bool = True) -> None:
    folder = root / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    if adapter:
        (folder / "adapter_model.safetensors").write_bytes(b"weights" * step)
        (folder / "adapter_config.json").write_text("{}", encoding="utf-8")


def write_manifest(root, entries) -> None:
    with open(root / "checkpoints.jsonl", "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


class TestDiscovery:
    def test_finds_checkpoints_in_step_order(self, tmp_path):
        for step in (300, 100, 200):
            make_checkpoint(tmp_path, step)
        found = discover_checkpoints(tmp_path, include_base=False)
        assert [c.step for c in found] == [100, 200, 300]

    def test_base_model_comes_first(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        found = discover_checkpoints(tmp_path, base_model="org/model", include_base=True)
        assert found[0].is_base
        assert found[0].step == 0

    def test_directories_without_adapters_are_skipped(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        make_checkpoint(tmp_path, 200, adapter=False)
        found = discover_checkpoints(tmp_path, include_base=False)
        assert [c.step for c in found] == [100]

    def test_unrelated_directories_are_ignored(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        (tmp_path / "logs").mkdir()
        (tmp_path / "reward-model").mkdir()
        found = discover_checkpoints(tmp_path, include_base=False)
        assert len(found) == 1

    def test_manifest_enriches_the_checkpoints(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        write_manifest(
            tmp_path,
            [{"step": 100, "epoch": 0.5, "stage": "sft", "train_loss": 1.2, "eval_loss": 1.4}],
        )
        found = discover_checkpoints(tmp_path, include_base=False)
        assert found[0].eval_loss == 1.4
        assert found[0].epoch == 0.5

    def test_a_resumed_run_overwrites_the_earlier_row(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        write_manifest(
            tmp_path,
            [
                {"step": 100, "eval_loss": 2.0},
                {"step": 100, "eval_loss": 1.1},
            ],
        )
        assert discover_checkpoints(tmp_path, include_base=False)[0].eval_loss == 1.1

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_checkpoints(tmp_path / "nope")


class TestFingerprint:
    def test_same_checkpoint_is_stable(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        ckpt = discover_checkpoints(tmp_path, include_base=False)[0]
        assert ckpt.fingerprint() == ckpt.fingerprint()

    def test_different_runs_do_not_collide(self, tmp_path):
        # Both runs write "checkpoint-100"; their cached results must stay apart.
        a, b = tmp_path / "run-a", tmp_path / "run-b"
        make_checkpoint(a, 100)
        make_checkpoint(b, 100)
        first = discover_checkpoints(a, include_base=False)[0]
        second = discover_checkpoints(b, include_base=False)[0]
        assert first.fingerprint() != second.fingerprint()


class TestResultCache:
    def _ckpt(self, tmp_path):
        make_checkpoint(tmp_path, 100)
        return discover_checkpoints(tmp_path, include_base=False)[0]

    def test_round_trip(self, tmp_path):
        cache = ResultCache(tmp_path / "reports")
        ckpt = self._ckpt(tmp_path)
        key = cache.key(ckpt, "val.jsonl", ["exact_match"], {"temperature": 0.0})

        assert cache.get(key) is None
        cache.put(key, {"metrics": {"exact_match": 0.5}})
        assert cache.get(key)["metrics"]["exact_match"] == 0.5

    def test_key_ignores_task_ordering(self, tmp_path):
        cache = ResultCache(tmp_path / "reports")
        ckpt = self._ckpt(tmp_path)
        decoding = {"temperature": 0.0}
        assert cache.key(ckpt, "v", ["a", "b"], decoding) == cache.key(
            ckpt, "v", ["b", "a"], decoding
        )

    def test_key_changes_with_decoding(self, tmp_path):
        cache = ResultCache(tmp_path / "reports")
        ckpt = self._ckpt(tmp_path)
        assert cache.key(ckpt, "v", ["a"], {"temperature": 0.0}) != cache.key(
            ckpt, "v", ["a"], {"temperature": 0.7}
        )

    def test_key_changes_with_dataset(self, tmp_path):
        cache = ResultCache(tmp_path / "reports")
        ckpt = self._ckpt(tmp_path)
        d = {"temperature": 0.0}
        assert cache.key(ckpt, "val.jsonl", ["a"], d) != cache.key(ckpt, "test.jsonl", ["a"], d)

    def test_disabled_cache_never_hits(self, tmp_path):
        cache = ResultCache(tmp_path / "reports", enabled=False)
        cache.put("k", {"metrics": {}})
        assert cache.get("k") is None

    def test_corrupt_entry_is_discarded(self, tmp_path):
        cache = ResultCache(tmp_path / "reports")
        (cache.root / "abc.json").write_text("{ truncated", encoding="utf-8")
        assert cache.get("abc") is None
        assert not (cache.root / "abc.json").exists()


def result_row(name, step, em, *, base=False, cached=False, seconds=10.0):
    return {
        "name": name,
        "step": step,
        "is_base": base,
        "stage": "base" if base else "sft",
        "metrics": {"exact_match": em, "length_ratio": 1.1},
        "num_examples": 100,
        "seconds": seconds,
        "from_cache": cached,
        "samples": [],
    }


class TestSummarise:
    def test_picks_the_best_and_the_delta(self):
        rows = [result_row("base", 0, 0.40, base=True), result_row("checkpoint-100", 100, 0.50)]
        summary = summarise(rows, ["exact_match", "length_ratio"])
        assert summary["best"]["name"] == "checkpoint-100"
        assert summary["delta"] == pytest.approx(0.10)
        assert summary["delta_pct"] == pytest.approx(25.0)

    def test_skips_non_directional_metrics_as_primary(self):
        rows = [result_row("checkpoint-100", 100, 0.5)]
        assert summarise(rows, ["length_ratio", "exact_match"])["primary_metric"] == "exact_match"

    def test_no_baseline_means_no_delta(self):
        summary = summarise([result_row("checkpoint-100", 100, 0.5)], ["exact_match"])
        assert summary["best"] is not None
        assert summary["delta"] is None

    def test_empty_results(self):
        assert summarise([], ["exact_match"])["best"] is None


class TestReport:
    def _cfg(self):
        cfg = PipelineConfig()
        cfg.eval.tasks = ["exact_match", "length_ratio"]
        return cfg

    def test_checkpoints_are_ordered_by_step(self):
        rows = [result_row("checkpoint-200", 200, 0.5), result_row("base", 0, 0.4, base=True)]
        report = build_report(self._cfg(), rows)
        assert [r["step"] for r in report["checkpoints"]] == [0, 200]

    def test_timing_counts_cache_hits(self):
        rows = [
            result_row("checkpoint-100", 100, 0.5, seconds=20.0),
            result_row("checkpoint-200", 200, 0.5, cached=True),
        ]
        timing = build_report(self._cfg(), rows, wall_seconds=25.0)["timing"]
        assert timing["evaluated"] == 1
        assert timing["from_cache"] == 1
        assert timing["estimated_seconds_saved"] == pytest.approx(20.0)

    def test_writes_every_artifact(self, tmp_path):
        report = build_report(self._cfg(), [result_row("checkpoint-100", 100, 0.5)])
        dashboard = tmp_path / "dash" / "runs.json"
        written = write_report(report, tmp_path / "reports", dashboard)

        assert (tmp_path / "reports" / "report.json").exists()
        assert (tmp_path / "reports" / "report.md").exists()
        assert dashboard.exists()
        assert len(written) == 4

    def test_dashboard_copy_is_optional(self, tmp_path):
        report = build_report(self._cfg(), [result_row("checkpoint-100", 100, 0.5)])
        written = write_report(report, tmp_path / "reports", None)
        assert len(written) == 3

    def test_markdown_has_a_row_per_checkpoint(self):
        rows = [result_row("base", 0, 0.4, base=True), result_row("checkpoint-100", 100, 0.5)]
        markdown = to_markdown(build_report(self._cfg(), rows))
        assert "| base | 0 |" in markdown
        assert "| checkpoint-100 | 100 |" in markdown
        assert "+25.0% vs base" in markdown


class TestBundledDashboardData:
    """The checked-in dashboard feed must match the schema the page expects."""

    def test_shape(self):
        with open("dashboard/data/runs.json", encoding="utf-8") as fh:
            report = json.load(fh)

        assert report["report_version"] >= 2
        assert report["tasks"]
        assert len(report["checkpoints"]) >= 15
        assert report["summary"]["best"]["name"]
        for row in report["checkpoints"]:
            assert set(report["tasks"]) <= set(row["metrics"])
