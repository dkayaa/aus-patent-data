"""Lexical metrics: ROUGE-L, token F1, containment."""

from __future__ import annotations

import re

from rouge_score import rouge_scorer

from schema import simple_tokenize

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_WS_RE = re.compile(r"\s+")


def normalize_for_containment(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def rouge_l_f1(reference: str, candidate: str) -> float:
    if not reference.strip() or not candidate.strip():
        return 0.0
    scores = _scorer.score(reference, candidate)
    return float(scores["rougeL"].fmeasure)


def token_f1_from_lists(ref: list[str], cand: list[str]) -> float:
    if not ref or not cand:
        return 0.0
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in cand:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(reference: str, candidate: str) -> float:
    return token_f1_from_lists(simple_tokenize(reference), simple_tokenize(candidate))


def best_span_f1(context: str, answer: str) -> float:
    """Max bag-of-tokens F1 of ``answer`` against a sliding window of ``context``."""
    if answer_contained(answer, context):
        return 1.0
    ctx = simple_tokenize(context)
    ans = simple_tokenize(answer)
    if not ctx or not ans:
        return 0.0
    max_win = min(len(ctx), max(4 * len(ans), 40))
    min_win = min(len(ans), max_win)
    if min_win < 1:
        return 0.0
    best = 0.0
    n = len(ctx)
    for win_len in range(min_win, max_win + 1):
        last = n - win_len + 1
        for start in range(last):
            score = token_f1_from_lists(ctx[start : start + win_len], ans)
            if score > best:
                best = score
                if best >= 1.0:
                    return 1.0
    return best


def answer_contained(answer: str, context: str) -> bool:
    a = normalize_for_containment(answer)
    c = normalize_for_containment(context)
    if not a:
        return False
    return a in c
