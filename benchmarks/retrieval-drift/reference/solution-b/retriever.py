"""
Reference Solution B — Explicit unit sphere projection & dynamic cosine similarity.

Fix Strategy:
1. Dynamically normalizes query and document embedding matrices to unit length.
2. Preserves full query tokens with hyphen-aware tokenization.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DenseRetriever:
    def __init__(self, corpus_path: str | pathlib.Path | None = None) -> None:
        if corpus_path is None:
            corpus_path = pathlib.Path(__file__).parent / "data" / "corpus.json"
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus: list[dict[str, Any]] = json.load(f)

        self.doc_ids = [d["id"] for d in self.corpus]
        self.doc_texts = [f"{d.get('title', '')} {d.get('text', '')}" for d in self.corpus]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w-]+\b",
        )
        self.doc_embeddings: np.ndarray | None = None
        self._index_corpus()

    def _index_corpus(self) -> None:
        raw_matrix = self.vectorizer.fit_transform(self.doc_texts).toarray()
        # Explicit unit normalization
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_embeddings = raw_matrix / norms

    def preprocess_query(self, query: str) -> str:
        # Lowercase and retain word characters and hyphens
        return " ".join(re.findall(r"[\w-]+", query.lower()))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        clean_q = self.preprocess_query(query)
        raw_q = self.vectorizer.transform([clean_q]).toarray()

        q_norm = np.linalg.norm(raw_q)
        if q_norm == 0 or self.doc_embeddings is None:
            return [
                {"id": self.doc_ids[i], "score": 0.0} for i in range(min(top_k, len(self.doc_ids)))
            ]

        # Compute cosine similarity
        scores = cosine_similarity(raw_q, self.doc_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "id": self.doc_ids[idx],
                    "score": float(scores[idx]),
                    "title": self.corpus[idx].get("title", ""),
                }
            )
        return results
