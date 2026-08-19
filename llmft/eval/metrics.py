"""Metric implementations.

Deliberately dependency-free (stdlib + numpy). `evaluate`/`rouge_score` pull in a
surprising amount of weight and, more importantly, their normalisation differs
subtly from what we want here, so the numbers stop being comparable across
versions. These are small enough to read and to unit test.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles, punctuation, spacing."""
    text = (text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalise(prediction) == normalise(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1. Handles the empty-string cases explicitly - the naive
    version returns 0 when both sides are empty, which is wrong."""
    pred_tokens = tokenise(prediction)
    ref_tokens = tokenise(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0

    precision = shared / len(pred_tokens)
    recall = shared / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence, two rows instead of the full table."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure. beta=1.2 matches the original ROUGE package default."""
    pred_tokens = tokenise(prediction)
    ref_tokens = tokenise(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def contains_answer(prediction: str, reference: str) -> float:
    """Lenient credit: the reference appears somewhere in the prediction.

    Useful for chatty checkpoints that answer correctly but wrap the answer in
    a sentence, which exact_match punishes without telling you anything.
    """
    ref = normalise(reference)
    return float(bool(ref) and ref in normalise(prediction))


def length_ratio(prediction: str, reference: str) -> float:
    """Prediction length over reference length. Not a quality score - it is the
    fastest way to spot a checkpoint that has started rambling or collapsed."""
    ref_len = len(tokenise(reference))
    if ref_len == 0:
        return 0.0
    return len(tokenise(prediction)) / ref_len


METRICS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
    "contains_answer": contains_answer,
    "length_ratio": length_ratio,
}

# Metrics where a higher number is not automatically better, so the dashboard
# knows not to crown a "best" checkpoint by them.
NON_DIRECTIONAL = {"length_ratio"}


def get_metric(name: str) -> Callable[[str, str], float]:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; available: {sorted(METRICS)}") from None


def score_batch(
    predictions: Sequence[str], references: Sequence[str], metric_names: Sequence[str]
) -> dict[str, float]:
    """Mean score per metric over a batch of prediction/reference pairs."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions/references length mismatch: {len(predictions)} vs {len(references)}"
        )
    if not predictions:
        return {name: 0.0 for name in metric_names}

    out: dict[str, float] = {}
    for name in metric_names:
        fn = get_metric(name)
        out[name] = sum(fn(p, r) for p, r in zip(predictions, references)) / len(predictions)
    return out
