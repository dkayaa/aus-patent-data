"""Semantic cosine similarity via sentence-transformers (Nomic Embed)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
# Nomic requires a task prefix. Mode 1 is document–document relatedness
# (claims vs abstract / justification / answer), not retrieval QA.
DEFAULT_PREFIX = "search_document: "


class SemanticScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_seq_length: int = 8192,
        batch_size: int = 32,
        prefix_a: str = DEFAULT_PREFIX,
        prefix_b: str = DEFAULT_PREFIX,
        trust_remote_code: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.prefix_a = prefix_a
        self.prefix_b = prefix_b

    @staticmethod
    def _prefixed(texts: Sequence[str], prefix: str) -> list[str]:
        if not prefix:
            return list(texts)
        out: list[str] = []
        for text in texts:
            if text.startswith(prefix):
                out.append(text)
            else:
                out.append(f"{prefix}{text}")
        return out

    def cosine_pair(self, text_a: str, text_b: str) -> float:
        scores = self.cosine_pairs([text_a], [text_b])
        return scores[0]

    def cosine_pairs(
        self, texts_a: Sequence[str], texts_b: Sequence[str]
    ) -> list[float]:
        if len(texts_a) != len(texts_b):
            raise ValueError("texts_a and texts_b must have the same length")
        if not texts_a:
            return []
        emb_a = self.model.encode(
            self._prefixed(texts_a, self.prefix_a),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb_b = self.model.encode(
            self._prefixed(texts_b, self.prefix_b),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        dots = np.sum(emb_a * emb_b, axis=1)
        return [float(x) for x in dots]
