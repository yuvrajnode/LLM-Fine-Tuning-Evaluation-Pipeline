from __future__ import annotations

import pytest

from llmft.data.formatting import TEMPLATES, get_template
from llmft.data.loaders import load_preference_records, load_sft_records, split_records
from llmft.data.preference import Candidate, build_preference_pairs


class TestTemplates:
    def test_context_changes_the_rendering(self):
        template = get_template("alpaca")
        assert "### Input:" in template.render("do the thing", "some context")
        assert "### Input:" not in template.render("do the thing")

    def test_blank_context_is_treated_as_absent(self):
        template = get_template("alpaca")
        assert template.render("q", "   ") == template.render("q", None)

    def test_render_full_is_prompt_plus_response(self):
        template = get_template("alpaca")
        prompt = template.render("q")
        assert template.render_full("q", "the answer") == prompt + "the answer"

    def test_prompt_is_a_prefix_of_the_full_text(self):
        # The eval harness relies on this: it renders only the prompt half and
        # expects the model to produce the rest.
        for template in TEMPLATES.values():
            full = template.render_full("q", "a", "ctx")
            assert full.startswith(template.render("q", "ctx"))

    def test_unknown_template_lists_the_options(self):
        with pytest.raises(KeyError, match="alpaca"):
            get_template("llama3")


class TestLoadSFT:
    def test_reads_and_renders(self, jsonl, data_cfg):
        path = jsonl(
            "t.jsonl",
            [{"instruction": "q1", "input": "", "output": "a1"}],
        )
        records, stats = load_sft_records(path, data_cfg)
        assert stats.kept == 1
        assert records[0].response == "a1"
        assert records[0].text.endswith("a1")

    def test_drops_rows_missing_a_field(self, jsonl, data_cfg):
        path = jsonl("t.jsonl", [{"instruction": "q"}, {"instruction": "q2", "output": "a"}])
        _, stats = load_sft_records(path, data_cfg)
        assert stats.dropped_missing == 1
        assert stats.kept == 1

    def test_drops_empty_values(self, jsonl, data_cfg):
        path = jsonl("t.jsonl", [{"instruction": "  ", "output": "a"}])
        _, stats = load_sft_records(path, data_cfg)
        assert stats.dropped_empty == 1

    def test_deduplicates(self, jsonl, data_cfg):
        row = {"instruction": "q", "output": "a"}
        path = jsonl("t.jsonl", [row, dict(row), dict(row)])
        records, stats = load_sft_records(path, data_cfg)
        assert len(records) == 1
        assert stats.dropped_duplicate == 2

    def test_deduplication_can_be_turned_off(self, jsonl, data_cfg):
        row = {"instruction": "q", "output": "a"}
        path = jsonl("t.jsonl", [row, dict(row)])
        records, _ = load_sft_records(path, data_cfg, deduplicate=False)
        assert len(records) == 2

    def test_limit_stops_early(self, jsonl, data_cfg):
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(20)]
        records, _ = load_sft_records(jsonl("t.jsonl", rows), data_cfg, limit=5)
        assert len(records) == 5

    def test_malformed_json_reports_the_line_number(self, tmp_path, data_cfg):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"instruction": "q", "output": "a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            load_sft_records(str(path), data_cfg)


class TestLoadPreference:
    def test_reads_triples(self, jsonl, data_cfg):
        path = jsonl(
            "p.jsonl",
            [{"instruction": "q", "chosen": "good", "rejected": "bad"}],
        )
        pairs = load_preference_records(path, data_cfg)
        assert pairs[0]["chosen"] == "good"
        assert "q" in pairs[0]["prompt"]

    def test_identical_sides_carry_no_signal(self, jsonl, data_cfg):
        path = jsonl("p.jsonl", [{"instruction": "q", "chosen": "same", "rejected": "same"}])
        assert load_preference_records(path, data_cfg) == []

    def test_missing_side_is_skipped(self, jsonl, data_cfg):
        path = jsonl("p.jsonl", [{"instruction": "q", "chosen": "good"}])
        assert load_preference_records(path, data_cfg) == []


class TestSplit:
    def test_split_is_disjoint_and_complete(self, jsonl, data_cfg):
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)]
        records, _ = load_sft_records(jsonl("t.jsonl", rows), data_cfg)
        train, held_out = split_records(records, eval_fraction=0.1, seed=7)

        assert len(train) + len(held_out) == len(records)
        train_texts = {r.text for r in train}
        assert not train_texts & {r.text for r in held_out}

    def test_split_is_deterministic(self, jsonl, data_cfg):
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(50)]
        records, _ = load_sft_records(jsonl("t.jsonl", rows), data_cfg)
        first = [r.text for r in split_records(records, seed=3)[1]]
        second = [r.text for r in split_records(records, seed=3)[1]]
        assert first == second

    def test_rejects_a_silly_fraction(self, jsonl, data_cfg):
        rows = [{"instruction": "q", "output": "a"}]
        records, _ = load_sft_records(jsonl("t.jsonl", rows), data_cfg)
        with pytest.raises(ValueError, match="eval_fraction"):
            split_records(records, eval_fraction=0.9)


class TestPreferencePairs:
    def test_orders_by_score(self):
        pairs = build_preference_pairs("p", [Candidate("worse", 0.1), Candidate("better", 0.9)])
        assert pairs[0].chosen == "better"
        assert pairs[0].rejected == "worse"

    def test_drops_near_ties(self):
        pairs = build_preference_pairs(
            "p", [Candidate("a", 0.50), Candidate("b", 0.52)], min_margin=0.05
        )
        assert pairs == []

    def test_caps_pairs_per_prompt(self):
        candidates = [Candidate(f"c{i}", i / 10) for i in range(8)]
        pairs = build_preference_pairs("p", candidates, max_pairs_per_prompt=3)
        assert len(pairs) == 3

    def test_keeps_the_widest_margins(self):
        candidates = [Candidate("low", 0.0), Candidate("mid", 0.5), Candidate("high", 1.0)]
        pairs = build_preference_pairs("p", candidates, max_pairs_per_prompt=1)
        assert pairs[0].margin == pytest.approx(1.0)

    def test_needs_at_least_two_candidates(self):
        assert build_preference_pairs("p", [Candidate("only", 1.0)]) == []

    def test_ignores_blank_candidates(self):
        pairs = build_preference_pairs("p", [Candidate("  ", 1.0), Candidate("real", 0.2)])
        assert pairs == []
