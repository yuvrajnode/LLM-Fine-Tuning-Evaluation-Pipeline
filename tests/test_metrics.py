from __future__ import annotations

import pytest

from llmft.eval.metrics import (
    contains_answer,
    exact_match,
    get_metric,
    length_ratio,
    normalise,
    rouge_l,
    score_batch,
    token_f1,
)


class TestNormalise:
    def test_lowercases_and_strips_punctuation(self):
        assert normalise("The Cat, sat!") == "cat sat"

    def test_collapses_whitespace(self):
        assert normalise("  a\t b \n c ") == "b c"  # "a" is an article

    def test_handles_none(self):
        assert normalise(None) == ""


class TestExactMatch:
    def test_ignores_case_and_punctuation(self):
        assert exact_match("Paris.", "paris") == 1.0

    def test_different_strings(self):
        assert exact_match("Paris", "London") == 0.0


class TestTokenF1:
    def test_identical(self):
        assert token_f1("a quick fox", "a quick fox") == 1.0

    def test_partial_overlap(self):
        # {quick, brown, fox} vs {quick, red, fox}: 2 shared, p = r = 2/3
        assert token_f1("quick brown fox", "quick red fox") == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert token_f1("cat", "dog") == 0.0

    def test_both_empty_is_a_match(self):
        # The naive implementation returns 0 here, which is wrong: predicting
        # nothing when nothing is expected is correct.
        assert token_f1("", "") == 1.0

    def test_one_side_empty(self):
        assert token_f1("", "something") == 0.0
        assert token_f1("something", "") == 0.0


class TestRougeL:
    def test_identical(self):
        assert rouge_l("the cat sat", "the cat sat") == 1.0

    def test_rewards_order(self):
        in_order = rouge_l("quick brown fox jumps", "quick brown fox leaps")
        shuffled = rouge_l("jumps fox brown quick", "quick brown fox leaps")
        assert in_order > shuffled

    def test_disjoint(self):
        assert rouge_l("alpha beta", "gamma delta") == 0.0

    def test_subsequence_need_not_be_contiguous(self):
        assert rouge_l("red big shiny ball", "red ball") > 0.5


class TestContainsAnswer:
    def test_finds_reference_inside_a_sentence(self):
        assert contains_answer("I think the answer is Paris, France.", "Paris") == 1.0

    def test_absent(self):
        assert contains_answer("I have no idea.", "Paris") == 0.0

    def test_empty_reference_is_not_a_free_point(self):
        assert contains_answer("anything at all", "") == 0.0


class TestLengthRatio:
    def test_equal_lengths(self):
        assert length_ratio("one two three", "four five six") == 1.0

    def test_rambling_prediction(self):
        assert length_ratio("one two three four", "one two") == 2.0

    def test_empty_reference(self):
        assert length_ratio("anything", "") == 0.0


class TestScoreBatch:
    def test_averages_across_examples(self):
        scores = score_batch(["yes", "no"], ["yes", "yes"], ["exact_match"])
        assert scores["exact_match"] == 0.5

    def test_empty_batch(self):
        assert score_batch([], [], ["exact_match", "rouge_l"]) == {
            "exact_match": 0.0,
            "rouge_l": 0.0,
        }

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            score_batch(["a"], ["a", "b"], ["exact_match"])

    def test_unknown_metric_names_the_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            get_metric("bleu")
