"""Faithfulness checking for ipc_reasoning via MiniCheck-Flan-T5-Large.

Additive and non-gating. Design (post-calibration + style audit, Aug 2026):

  1. Atomicise the justification.
  2. **Decide first, score second:** drop META atoms (alignment / bridge
     markers, or zero claim technical terms) *before* MiniCheck. Do not
     post-hoc exclude only the atoms that already failed — that selects on
     the outcome.
  3. Score remaining atoms against the **combined** claims+definition doc.
  4. Three-way band on P(combined):
       SUPPORTED   P >= support_high  (default 0.7)
       UNDECIDED   support_low <= P < support_high  (default 0.3–0.7)
       UNSUPPORTED P < support_low   (default 0.3)
     The mid band is where MiniCheck is unsure; report it, don't force binary.

Faithfulness ≠ correct IPC reasoning (wrong-bridge needs expert audit).

Model: lytang/MiniCheck-Flan-T5-Large (MIT). Do NOT use Bespoke-MiniCheck-7B.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "ckpts"
DOC_TOKEN_WARN = 32_000
DEFAULT_SUPPORT_HIGH = 0.7
DEFAULT_SUPPORT_LOW = 0.3
_MIN_ATOMIC_WORDS = 6

_ATOMIC_SPLIT = re.compile(
    r"\s*;\s+"
    r"|\s+[—–]\s+"
    r"|(?<=,)\s+(?=which\b)"
    r"|(?<=,)\s+(?=wherein\b)"
    r"|(?<=,)\s+(?=whereby\b)"
    r"|(?<=,)\s+(?=thereby\b)"
    r"|(?<=,)\s+(?=consistent with\b)"
    r"|(?<=,)\s+(?=directly (?:maps|corresponding|corresponds)\b)"
    r"|(?<=,)\s+(?=further (?:aligns|aligning)\b)",
    re.IGNORECASE,
)

# Bridge / alignment style — unverifiable by construction. Checked on every
# atom *before* MiniCheck (not only on failures).
_META_MARKERS = (
    "aligns with",
    "aligns precisely",
    "aligning with",
    "falls within",
    "falls squarely",
    "consistent with",
    "maps to",
    "maps directly",
    "directly maps",
    "corresponds to",
    "corresponding to",
    "directly corresponding",
    "as defined",
    "as characterised",
    "as characterized",
    "within the definition",
    "within the scope",
    "scope of the definition",
    "definition's",
    "definition of",
    "the definition",
    "which constitutes",
    "further aligns",
    "mirrors the definition",
    "fulfilling the",
    "as encompassed by the definition",
)


def _ensure_nltk_data() -> None:
    venv_data = REPO_ROOT / ".venv" / "nltk_data"
    if venv_data.is_dir():
        os.environ.setdefault("NLTK_DATA", str(venv_data))
        try:
            import nltk

            if str(venv_data) not in nltk.data.path:
                nltk.data.path.insert(0, str(venv_data))
        except ImportError:
            pass


def split_sentences(text: str) -> list[str]:
    _ensure_nltk_data()
    from nltk.tokenize import sent_tokenize

    cleaned = (text or "").strip()
    if not cleaned:
        return []
    return [s.strip() for s in sent_tokenize(cleaned) if s.strip()]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def atomicize_justification(text: str) -> list[str]:
    """Sentence-split then break long compounds into short atomic statements."""
    atoms: list[str] = []
    for sent in split_sentences(text):
        parts = [p.strip(" ,") for p in _ATOMIC_SPLIT.split(sent) if p and p.strip(" ,")]
        if len(parts) <= 1:
            atoms.append(sent)
            continue
        buf = ""
        for part in parts:
            if buf and _word_count(part) < _MIN_ATOMIC_WORDS:
                buf = f"{buf} {part}".strip()
                continue
            if buf:
                atoms.append(buf)
            buf = part
        if buf:
            if atoms and _word_count(buf) < _MIN_ATOMIC_WORDS:
                atoms[-1] = f"{atoms[-1]} {buf}".strip()
            else:
                atoms.append(buf)
    return atoms


def has_meta_marker(text: str) -> bool:
    sl = (text or "").lower()
    return any(m in sl for m in _META_MARKERS)


def classify_meta(sentence: str, *, n_claim_terms: int) -> str | None:
    """Return a meta reason if this atom should be skipped before MiniCheck.

    Reasons: ``alignment`` (bridge markers) or ``empty`` (no claim terms).
    """
    if has_meta_marker(sentence):
        return "alignment"
    if n_claim_terms <= 0:
        return "empty"
    return None


def band_from_prob(
    prob: float, *, support_low: float, support_high: float
) -> str:
    if prob >= support_high:
        return "SUPPORTED"
    if prob < support_low:
        return "UNSUPPORTED"
    return "UNDECIDED"


def combined_document(*, claims: str, definition: str) -> str:
    return (
        "=== PATENT CLAIMS ===\n"
        f"{claims.strip()}\n\n"
        "=== WIPO IPC DEFINITION ===\n"
        f"{definition.strip()}"
    )


@dataclass
class SentenceFaithfulness:
    sentence: str
    state: str  # META | SUPPORTED | UNDECIDED | UNSUPPORTED
    meta_reason: str | None
    support_combined: int | None
    support_combined_prob: float | None
    n_claim_terms: int
    scored: bool
    support_claims: int | None = None
    support_claims_prob: float | None = None
    support_definition: int | None = None
    support_definition_prob: float | None = None

    @property
    def supported(self) -> bool:
        return self.state == "SUPPORTED"

    @property
    def undecided(self) -> bool:
        return self.state == "UNDECIDED"


@dataclass
class RecordFaithfulness:
    application_number: str
    n_sentences: int
    n_scored: int
    n_meta: int
    n_supported: int
    n_undecided: int
    n_unsupported: int
    # Among scored atoms only (meta excluded from denominator).
    faithfulness_rate: float | None
    undecided_rate: float | None
    min_prob_combined: float | None
    support_low: float
    support_high: float
    meta_sentences: list[str]
    unsupported_sentences: list[str]
    undecided_sentences: list[str]
    sentences: list[SentenceFaithfulness]


class FaithfulnessScorer:
    """MiniCheck-Flan-T5-Large wrapper for ipc_reasoning justifications."""

    def __init__(
        self,
        *,
        cache_dir: Path | str | None = None,
        batch_size: int = 8,
        support_high: float = DEFAULT_SUPPORT_HIGH,
        support_low: float = DEFAULT_SUPPORT_LOW,
        score_halves: bool = False,
        terms: Any | None = None,
    ) -> None:
        _ensure_nltk_data()
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        from minicheck.minicheck import MiniCheck

        from terms_coverage import TermsCoverageScorer

        cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading MiniCheck-Flan-T5-Large (cache_dir=%s); MPS/CPU via device_map=auto",
            cache,
        )
        self.scorer = MiniCheck(
            model_name="flan-t5-large",
            cache_dir=str(cache),
            batch_size=batch_size,
        )
        self.support_high = float(support_high)
        self.support_low = float(support_low)
        self.score_halves = bool(score_halves)
        self.terms = terms if terms is not None else TermsCoverageScorer()
        self._approx_tok = None

    def _approx_tokens(self, text: str) -> int:
        if self._approx_tok is None:
            try:
                from transformers import AutoTokenizer

                self._approx_tok = AutoTokenizer.from_pretrained(
                    "lytang/MiniCheck-Flan-T5-Large",
                    local_files_only=True,
                )
            except Exception:  # noqa: BLE001
                self._approx_tok = False
        if self._approx_tok is False:
            return max(1, len(text) // 4)
        return len(self._approx_tok.encode(text, add_special_tokens=False))

    def _score_pairs(
        self, docs: Sequence[str], claims: Sequence[str]
    ) -> tuple[list[int], list[float]]:
        if not claims:
            return [], []
        pred_label, raw_prob, _, _ = self.scorer.score(
            docs=list(docs), claims=list(claims)
        )
        return [int(x) for x in pred_label], [float(x) for x in raw_prob]

    def score_justification(
        self,
        *,
        application_number: str,
        justification: str,
        claims: str,
        definition: str,
    ) -> RecordFaithfulness:
        empty = RecordFaithfulness(
            application_number=application_number,
            n_sentences=0,
            n_scored=0,
            n_meta=0,
            n_supported=0,
            n_undecided=0,
            n_unsupported=0,
            faithfulness_rate=None,
            undecided_rate=None,
            min_prob_combined=None,
            support_low=self.support_low,
            support_high=self.support_high,
            meta_sentences=[],
            unsupported_sentences=[],
            undecided_sentences=[],
            sentences=[],
        )
        sents = atomicize_justification(justification)
        if not sents:
            return empty

        claims_doc = (claims or "").strip()
        def_doc = (definition or "").strip()
        comb_doc = combined_document(claims=claims_doc, definition=def_doc)

        n_tok = self._approx_tokens(comb_doc)
        if n_tok > DOC_TOKEN_WARN:
            logger.warning(
                "faithfulness doc exceeds 32K est. tokens: app=%s doc=combined "
                "tokens≈%d (not chunking; MiniCheck library handles long inputs)",
                application_number,
                n_tok,
            )

        # Decide first: classify META before any MiniCheck call.
        meta_flags: list[str | None] = []
        n_claim_terms_list: list[int] = []
        to_score: list[str] = []
        score_index: list[int] = []
        for sent in sents:
            n_terms = int(self.terms.claim_term_hits(claims_doc, sent))
            n_claim_terms_list.append(n_terms)
            reason = classify_meta(sent, n_claim_terms=n_terms)
            meta_flags.append(reason)
            if reason is None:
                score_index.append(len(to_score))
                to_score.append(sent)
            else:
                score_index.append(-1)

        lab_x: list[int] = []
        prob_x: list[float] = []
        lab_c_map: dict[int, int | None] = {}
        prob_c_map: dict[int, float | None] = {}
        lab_d_map: dict[int, int | None] = {}
        prob_d_map: dict[int, float | None] = {}

        if to_score:
            n_s = len(to_score)
            lab_x, prob_x = self._score_pairs([comb_doc] * n_s, to_score)
            if self.score_halves:
                lc, pc = self._score_pairs([claims_doc] * n_s, to_score)
                ld, pd = self._score_pairs([def_doc] * n_s, to_score)
                for j in range(n_s):
                    lab_c_map[j] = lc[j]
                    prob_c_map[j] = pc[j]
                    lab_d_map[j] = ld[j]
                    prob_d_map[j] = pd[j]

        details: list[SentenceFaithfulness] = []
        n_meta = n_supported = n_undecided = n_unsupported = 0
        meta_sents: list[str] = []
        unsupported_sents: list[str] = []
        undecided_sents: list[str] = []
        scored_probs: list[float] = []

        for i, sent in enumerate(sents):
            reason = meta_flags[i]
            n_terms = n_claim_terms_list[i]
            if reason is not None:
                n_meta += 1
                meta_sents.append(sent)
                details.append(
                    SentenceFaithfulness(
                        sentence=sent,
                        state="META",
                        meta_reason=reason,
                        support_combined=None,
                        support_combined_prob=None,
                        n_claim_terms=n_terms,
                        scored=False,
                    )
                )
                continue

            j = score_index[i]
            prob = float(prob_x[j])
            lab = int(lab_x[j])
            scored_probs.append(prob)
            band = band_from_prob(
                prob, support_low=self.support_low, support_high=self.support_high
            )
            if band == "SUPPORTED":
                n_supported += 1
            elif band == "UNDECIDED":
                n_undecided += 1
                undecided_sents.append(sent)
            else:
                n_unsupported += 1
                unsupported_sents.append(sent)

            details.append(
                SentenceFaithfulness(
                    sentence=sent,
                    state=band,
                    meta_reason=None,
                    support_combined=lab,
                    support_combined_prob=prob,
                    n_claim_terms=n_terms,
                    scored=True,
                    support_claims=lab_c_map.get(j),
                    support_claims_prob=prob_c_map.get(j),
                    support_definition=lab_d_map.get(j),
                    support_definition_prob=prob_d_map.get(j),
                )
            )

        n_scored = n_supported + n_undecided + n_unsupported
        rate = float(n_supported) / float(n_scored) if n_scored else None
        und_rate = float(n_undecided) / float(n_scored) if n_scored else None
        return RecordFaithfulness(
            application_number=application_number,
            n_sentences=len(sents),
            n_scored=n_scored,
            n_meta=n_meta,
            n_supported=n_supported,
            n_undecided=n_undecided,
            n_unsupported=n_unsupported,
            faithfulness_rate=rate,
            undecided_rate=und_rate,
            min_prob_combined=min(scored_probs) if scored_probs else None,
            support_low=self.support_low,
            support_high=self.support_high,
            meta_sentences=meta_sents,
            unsupported_sentences=unsupported_sents,
            undecided_sentences=undecided_sents,
            sentences=details,
        )

    def summary_dict(self, result: RecordFaithfulness) -> dict[str, Any]:
        return {
            "faithfulness_model": "lytang/MiniCheck-Flan-T5-Large",
            "faithfulness_support_low": result.support_low,
            "faithfulness_support_high": result.support_high,
            "n_sentences": result.n_sentences,
            "n_scored": result.n_scored,
            "n_meta": result.n_meta,
            "n_supported": result.n_supported,
            "n_undecided": result.n_undecided,
            "n_unsupported": result.n_unsupported,
            "faithfulness_rate": result.faithfulness_rate,
            "undecided_rate": result.undecided_rate,
            "min_prob_combined": result.min_prob_combined,
            "meta_sentences": list(result.meta_sentences),
            "unsupported_sentences": list(result.unsupported_sentences),
            "undecided_sentences": list(result.undecided_sentences),
        }

    def sentence_detail_dict(self, result: RecordFaithfulness) -> dict[str, Any]:
        return {
            "application_number": result.application_number,
            "support_low": result.support_low,
            "support_high": result.support_high,
            "sentences": [
                {
                    "sentence": s.sentence,
                    "state": s.state,
                    "meta_reason": s.meta_reason,
                    "scored": s.scored,
                    "support_combined": s.support_combined,
                    "support_combined_prob": s.support_combined_prob,
                    "n_claim_terms": s.n_claim_terms,
                    "support_claims": s.support_claims,
                    "support_claims_prob": s.support_claims_prob,
                    "support_definition": s.support_definition,
                    "support_definition_prob": s.support_definition_prob,
                }
                for s in result.sentences
            ],
        }
