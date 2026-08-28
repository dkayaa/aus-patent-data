"""Semantic cosine similarity via configurable sentence-transformers models.

Chunking fallback (only when source exceeds model.max_seq_length):
  AlignScore (arXiv 2305.16739) chunks sources into ~350-token overlapping
  windows and takes the max per output sentence. Patent-CE (arXiv 2505.11095)
  chose Longformer at 4096 because patent claim sets average over 1000 tokens.
  We embed whole when it fits; otherwise overlapping windows (~25% overlap),
  report cosine_max (used for pass/fail) and cosine_mean, plus n_chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = (
    REPO_ROOT / "dataset-validation" / "config" / "embedding_models.yaml"
)


@dataclass(frozen=True)
class EmbeddingModelSpec:
    key: str
    model_name: str
    expected_max_seq_length: int
    trust_remote_code: bool
    document_prefix: str | None
    notes: str = ""


def load_embedding_registry(path: Path | None = None) -> dict[str, EmbeddingModelSpec]:
    reg_path = path or DEFAULT_REGISTRY
    with reg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    models = raw.get("models") or {}
    out: dict[str, EmbeddingModelSpec] = {}
    for key, cfg in models.items():
        out[str(key)] = EmbeddingModelSpec(
            key=str(key),
            model_name=str(cfg["model_name"]),
            expected_max_seq_length=int(cfg["expected_max_seq_length"]),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            document_prefix=cfg.get("document_prefix"),
            notes=str(cfg.get("notes") or ""),
        )
    if not out:
        raise ValueError(f"No embedding models registered in {reg_path}")
    return out


class SemanticScorer:
    """Encode text pairs and score cosine similarity with optional chunking."""

    def __init__(
        self,
        spec: EmbeddingModelSpec,
        *,
        batch_size: int = 32,
        chunk_overlap_frac: float = 0.25,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.batch_size = batch_size
        self.chunk_overlap_frac = float(chunk_overlap_frac)

        load_kwargs: dict[str, Any] = {}
        if spec.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        self.model = SentenceTransformer(spec.model_name, **load_kwargs)

        runtime_max = int(self.model.max_seq_length)
        expected = int(spec.expected_max_seq_length)

        # Granite r2 is ModernBERT at 8192; if sentence-transformers reports a
        # lower default, set it explicitly and log.
        if runtime_max < expected:
            logger.warning(
                "embedding model %s runtime max_seq_length=%d < expected=%d; "
                "setting explicitly to %d",
                spec.key,
                runtime_max,
                expected,
                expected,
            )
            self.model.max_seq_length = expected
            runtime_max = int(self.model.max_seq_length)

        if runtime_max != expected:
            raise RuntimeError(
                f"Embedding model {spec.key!r} ({spec.model_name}): "
                f"runtime max_seq_length={runtime_max} does not match "
                f"registry expected_max_seq_length={expected}. "
                f"Refusing to score with a mismatched window "
                f"(this is the MiniLM 512-vs-256 class of bug)."
            )

        self.max_seq_length = runtime_max
        logger.info(
            "Loaded embedding model key=%s name=%s max_seq_length=%d prefix=%r",
            spec.key,
            spec.model_name,
            self.max_seq_length,
            spec.document_prefix,
        )

    @property
    def model_key(self) -> str:
        return self.spec.key

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    def _prefix(self, text: str) -> str:
        p = self.spec.document_prefix
        if not p:
            return text
        # Avoid double-prefixing if caller already applied it.
        if text.startswith(p):
            return text
        return f"{p}{text}"

    def _prefix_token_budget(self) -> int:
        """Tokens consumed by document_prefix + special tokens, reserved from windows."""
        tok = getattr(self.model, "tokenizer", None)
        special = 2
        if tok is None:
            p = self.spec.document_prefix or ""
            return special + (len(p.split()) if p else 0)
        prev = getattr(tok, "model_max_length", None)
        try:
            # Avoid "sequence longer than max" warnings while measuring.
            tok.model_max_length = int(1e9)
            p = self.spec.document_prefix or ""
            prefix_ids = tok.encode(p, add_special_tokens=False) if p else []
            return special + len(prefix_ids)
        finally:
            if prev is not None:
                tok.model_max_length = prev

    def _encode_ids(self, text: str) -> list[int]:
        tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            return []
        prev = getattr(tok, "model_max_length", None)
        try:
            tok.model_max_length = int(1e9)
            return list(tok.encode(text, add_special_tokens=False))
        finally:
            if prev is not None:
                tok.model_max_length = prev

    def _chunk_text(self, text: str) -> list[str]:
        """Overlapping windows sized to max_seq_length with ~25% overlap.

        Uses the model's tokenizer for window boundaries when available.
        Window size reserves space for document_prefix + special tokens so the
        prefixed encode path never exceeds max_seq_length.
        """
        tok = getattr(self.model, "tokenizer", None)
        reserve = self._prefix_token_budget()
        max_len = self.max_seq_length
        window = max(8, max_len - reserve)

        if tok is None:
            words = text.split()
            if len(words) <= window:
                return [text]
            stride = max(1, int(window * (1.0 - self.chunk_overlap_frac)))
            chunks: list[str] = []
            for start in range(0, len(words), stride):
                piece = words[start : start + window]
                if not piece:
                    break
                chunks.append(" ".join(piece))
                if start + window >= len(words):
                    break
            return chunks or [text]

        ids = self._encode_ids(text)
        if len(ids) <= window:
            return [text]
        stride = max(1, int(window * (1.0 - self.chunk_overlap_frac)))
        chunks = []
        for start in range(0, len(ids), stride):
            piece = ids[start : start + window]
            if not piece:
                break
            chunks.append(tok.decode(piece, skip_special_tokens=True))
            if start + window >= len(ids):
                break
        return chunks or [text]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        # Always L2-normalise. Granite produces unnormalised vectors; cosine
        # via dot product requires unit vectors. Be explicit rather than
        # relying on a library default.
        prefixed = [self._prefix(t) for t in texts]
        return self.model.encode(
            list(prefixed),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def score_pair(self, source: str, output: str) -> dict[str, Any]:
        """Score source↔output.

        Whole-document encode when source fits; otherwise chunk source and
        take cosine_max against the (whole) output embedding.
        """
        source = source or ""
        output = output or ""
        if not source.strip() or not output.strip():
            return {
                "embedding_model": self.spec.key,
                "embedding_model_id": self.spec.model_name,
                "max_seq_length": self.max_seq_length,
                "semantic_cosine": 0.0,
                "semantic_cosine_max": 0.0,
                "semantic_cosine_mean": 0.0,
                "n_chunks": 0,
                "chunked": False,
            }

        chunks = self._chunk_text(source)
        n_chunks = len(chunks)
        chunked = n_chunks > 1

        out_emb = self._encode([output])[0]
        chunk_embs = self._encode(chunks)
        # Normalised → cosine = dot product.
        dots = chunk_embs @ out_emb
        cos_max = float(np.max(dots))
        cos_mean = float(np.mean(dots))

        return {
            "embedding_model": self.spec.key,
            "embedding_model_id": self.spec.model_name,
            "max_seq_length": self.max_seq_length,
            # semantic_cosine == cosine_max for pass/fail continuity.
            "semantic_cosine": cos_max,
            "semantic_cosine_max": cos_max,
            "semantic_cosine_mean": cos_mean,
            "n_chunks": n_chunks,
            "chunked": chunked,
        }

    def cosine_pair(self, text_a: str, text_b: str) -> float:
        """Backward-compatible single float (cosine_max)."""
        return float(self.score_pair(text_a, text_b)["semantic_cosine_max"])

    def cosine_pairs(
        self, texts_a: Sequence[str], texts_b: Sequence[str]
    ) -> list[float]:
        if len(texts_a) != len(texts_b):
            raise ValueError("texts_a and texts_b must have the same length")
        return [self.cosine_pair(a, b) for a, b in zip(texts_a, texts_b)]
