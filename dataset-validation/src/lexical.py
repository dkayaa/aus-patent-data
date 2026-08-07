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


def token_f1(reference: str, candidate: str) -> float:
    ref = simple_tokenize(reference)
    cand = simple_tokenize(candidate)
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


def answer_contained(answer: str, context: str) -> bool:
    a = normalize_for_containment(answer)
    c = normalize_for_containment(context)
    if not a:
        return False
    return a in c
