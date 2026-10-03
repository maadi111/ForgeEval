"""
Semantic Document Retriever for Technical Documentation.

Production Symptom:
Offline unit tests on historical cloud documentation achieve high recall, but
retrieval performance drops severely on recently added MLOps/Kubernetes technical queries.
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

        # TF-IDF embedding vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            norm=None,  # BUG 1 (intentional): Unnormalized embeddings!
        )
        self.doc_embeddings: np.ndarray | None = None
        self._index_corpus()

    def _index_corpus(self) -> None:
        matrix = self.vectorizer.fit_transform(self.doc_texts)
        # BUG 1 (cont.): Stores raw unnormalized matrix
        self.doc_embeddings = matrix.toarray()

    def preprocess_query(self, query: str) -> str:
        """Sanitize query string.

        BUG 2 (intentional): Aggressively strips compound hyphens and truncates
        the query to 4 words. While short historical queries survive, compound
        domain queries lose critical discriminative terms!
        """
        # Lowercase and strip punctuation
        cleaned = re.sub(r"[^\w\s]", " ", query.lower())
        tokens = cleaned.split()
        # Truncating to 4 words destroys technical compound queries
        truncated = tokens[:4]
        return " ".join(truncated)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top_k most relevant documents for a query."""
        clean_q = self.preprocess_query(query)
        q_vec = self.vectorizer.transform([clean_q]).toarray()[0]

        if np.all(q_vec == 0) or self.doc_embeddings is None:
            # Fallback
            return [
                {"id": self.doc_ids[i], "score": 0.0} for i in range(min(top_k, len(self.doc_ids)))
            ]

        # BUG 1: Computes raw unnormalized dot product
        # Longer documents with many tokens have huge norms and dominate similarity!
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
