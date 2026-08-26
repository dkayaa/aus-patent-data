"""Terms Coverage metric (additive; not a pass/fail gate).

terms_coverage = (# technical terms from input claims that also appear in
                   the output) / (# technical terms in the input claims)

Term extraction is deliberately simple and deterministic: tokenise, drop
stopwords + patent boilerplate, keep unigrams and bigrams, deduplicate.
Match case-insensitively after light normalisation.

Precedent: Zuo et al., PatentEval, NAACL 2024, Table 3 — on claims-to-abstract
generation, Terms Coverage reached Kendall tau 0.287 against patent expert
judgement, the highest of six metrics tested (above two BERT-for-Patents
semantic similarity variants at 0.266 and 0.256, and FactGraph at 0.065).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOILERPLATE = (
    REPO_ROOT / "dataset-validation" / "config" / "terms_boilerplate.yaml"
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)


def _norm(text: Any) -> str:
    return " ".join(str(text).lower().split())


def _light_stem(token: str) -> str:
    """Very light plural fold so module/modules still match."""
    t = token.lower()
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("sses", "xes", "zes", "ches", "shes")):
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


class TermsCoverageScorer:
    def __init__(self, boilerplate_path: Path | None = None) -> None:
        path = boilerplate_path or DEFAULT_BOILERPLATE
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        stops = { _norm(s) for s in (raw.get("stopwords") or []) if str(s).strip() }
        boiler = { _norm(s) for s in (raw.get("boilerplate") or []) if str(s).strip() }
        self._drop = stops | boiler
        # Multi-word boilerplate phrases (already normalised).
        self._phrases = sorted(
            (p for p in boiler if " " in p),
            key=len,
            reverse=True,
        )

    def _strip_phrases(self, text: str) -> str:
        t = _norm(text)
        for phrase in self._phrases:
            t = t.replace(phrase, " ")
        return t

    def extract_terms(self, text: str) -> list[str]:
        cleaned = self._strip_phrases(text)
        tokens = [m.group(0).lower() for m in _WORD_RE.finditer(cleaned)]
        tokens = [t for t in tokens if t not in self._drop and len(t) > 1]
        terms: list[str] = []
        seen: set[str] = set()
        # Unigrams
        for t in tokens:
            if t not in seen:
                seen.add(t)
                terms.append(t)
        # Bigrams
        for a, b in zip(tokens, tokens[1:]):
            bigram = f"{a} {b}"
            if a in self._drop or b in self._drop:
                continue
            if bigram in self._drop:
                continue
            if bigram not in seen:
                seen.add(bigram)
                terms.append(bigram)
        return terms

    def _term_in_text(self, term: str, out_norm: str, out_terms: set[str]) -> bool:
        if " " in term:
            if term in out_norm or term in out_terms:
                return True
            parts = term.split()
            folded = " ".join(_light_stem(p) for p in parts)
            out_folded = " ".join(_light_stem(w) for w in out_norm.split())
            return folded in out_folded or folded in {
                " ".join(_light_stem(x) for x in t.split()) for t in out_terms if " " in t
            }
        if term in out_terms or re.search(rf"\b{re.escape(term)}\b", out_norm) is not None:
            return True
        out_unigram_stems = {_light_stem(t) for t in out_terms if " " not in t}
        stem = _light_stem(term)
        if stem in out_unigram_stems:
            return True
        if re.search(rf"\b{re.escape(stem)}s?\b", out_norm) is not None:
            return True
        return False

    def claim_term_hits(self, claims_text: str, sentence: str) -> int:
        """Count how many claim technical terms appear in ``sentence``.

        Used as a per-atom empty-meta screen for ipc_reasoning faithfulness:
        a justification fragment with zero claim terms is alignment fluff.
        """
        input_terms = self.extract_terms(claims_text or "")
        if not input_terms:
            return 0
        out_norm = _norm(sentence or "")
        out_terms = set(self.extract_terms(sentence or ""))
        return sum(
            1 for term in input_terms if self._term_in_text(term, out_norm, out_terms)
        )

    def score(self, claims_text: str, output_text: str) -> dict[str, Any]:
        input_terms = self.extract_terms(claims_text or "")
        if not input_terms:
            return {
                "terms_coverage": None,
                "terms_n_input": 0,
                "terms_n_matched": 0,
                "terms_n_unmatched": 0,
                "terms_unmatched": [],
            }
        out_norm = _norm(output_text or "")
        out_terms = set(self.extract_terms(output_text or ""))
        matched: list[str] = []
        unmatched: list[str] = []
        for term in input_terms:
            if self._term_in_text(term, out_norm, out_terms):
                matched.append(term)
            else:
                unmatched.append(term)
        n_in = len(input_terms)
        n_matched = len(matched)
        return {
            "terms_coverage": float(n_matched) / float(n_in) if n_in else None,
            "terms_n_input": n_in,
            "terms_n_matched": n_matched,
            "terms_n_unmatched": len(unmatched),
            # Cap list size in persisted records to keep shards readable.
            "terms_unmatched": unmatched[:50],
        }
