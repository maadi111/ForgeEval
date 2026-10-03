# Hidden Test Rationale — B2 Retrieval Drift

This document details why each hidden test exists in the Retrieval Drift benchmark, and what failure modes it detects.

---

## 1. `test_drift_recall.py`

### `test_drifted_domain_recall_at_5`

**Why it exists:**  
Offline retrieval evaluation on historical logs creates a false sense of security. When a service expands to a new vertical or terminology shifts, uncalibrated embeddings and aggressive token truncation cause relevant documents to drop outside the top-K window.

**What it catches:**  
- Severe recall collapse under covariate and domain shift.
- Truncation that strips domain-specific keywords.
- Lack of embedding alignment between new queries and new documents.

---

## 2. `test_embedding_norm_invariance.py`

### `test_scores_are_bounded_like_cosine`

**Why it exists:**  
Cosine similarity should be bounded between $[-1.0, 1.0]$. An unnormalized inner-product search (`query @ docs.T`) skews heavily toward verbose documents with high L2 norms, burying short, precise documents.

**What it catches:**  
- Unnormalized document/query vector embeddings.
- Length-biased ranking artifacts.

---

## 3. `test_negative_controls.py`

### `test_nc1_constant_retriever_fails` & `test_nc2_random_retriever_fails`

**Why it exists:**  
Verifies that dummy shortcuts (e.g. always returning a popular document or random guessing) fail the benchmark evaluation criteria.
