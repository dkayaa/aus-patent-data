"""Semantic cosine similarity via sentence-transformers MiniLM."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class SemanticScorer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        max_seq_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

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
        # Encode separately then row-wise cosine (handles long batches).
        emb_a = self.model.encode(
            list(texts_a),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb_b = self.model.encode(
            list(texts_b),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # With normalized embeddings, cosine = dot product.
        dots = np.sum(emb_a * emb_b, axis=1)
        return [float(x) for x in dots]
