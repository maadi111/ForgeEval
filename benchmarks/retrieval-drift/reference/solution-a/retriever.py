"""
Reference Solution A — L2-normalized cosine similarity & compound token preservation.

Fix Strategy:
1. Enables L2 normalization on TF-IDF embeddings, ensuring similarity scores reflect
   semantic angle rather than document token length.
2. Removes aggressive 4-word truncation, preserving domain-specific terminology
   and hyphenated identifiers.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseRetriever:
    def __init__(self, corpus_path: str | pathlib.Path | None = None) -> None:
        if corpus_path is None:
            corpus_path = pathlib.Path(__file__).parent / "data" / "corpus.json"
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus: list[dict[str, Any]] = json.load(f)

        self.doc_ids = [d["id"] for d in self.corpus]
        self.doc_texts = [f"{d.get('title', '')} {d.get('text', '')}" for d in self.corpus]

        # FIX 1: L2 normalization + sublinear TF
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            norm="l2",
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w-]+\b",  # Retain hyphens in compound words like gpu-operator
        )
        self.doc_embeddings: np.ndarray | None = None
        self._index_corpus()

    def _index_corpus(self) -> None:
        matrix = self.vectorizer.fit_transform(self.doc_texts)
        self.doc_embeddings = matrix.toarray()

    def preprocess_query(self, query: str) -> str:
        """FIX 2: Preserve full query and technical compound tokens."""
        # Normalize whitespace and lowercase, retaining alphanumeric and hyphen
        cleaned = re.sub(r"[^\w\s-]", " ", query.lower())
        return " ".join(cleaned.split())

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        clean_q = self.preprocess_query(query)
        q_vec = self.vectorizer.transform([clean_q]).toarray()[0]

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0 or self.doc_embeddings is None:
            return [
                {"id": self.doc_ids[i], "score": 0.0} for i in range(min(top_k, len(self.doc_ids)))
            ]

        # Vector is already L2-normalized by TfidfVectorizer(norm='l2')
        scores = np.dot(self.doc_embeddings, q_vec)
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
