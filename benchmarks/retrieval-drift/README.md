# Retrieval Drift Benchmark

## Candidate Symptom

Our semantic document search engine passes historical regression benchmarks with strong Recall@5 (> 85%), but after a recent ingestion of new domain documentation and updated search queries, users report that relevant documents are missing from top search results. Retrieval quality on drifted queries has severely degraded, while offline evaluations on historical queries remain falsely confident.

## Candidate Objective

Diagnose and repair the embedding retrieval and indexing pipeline so that the retriever generalizes effectively across domain shifts and maintains robust Recall@5 and Precision@5 on both historical and drifted distributions without exceeding latency budgets.

## Planned Evaluation Criteria

- Recall@5 on historical domain queries
- Recall@5 and Precision@5 under distribution shift (drift robustness)
- Embedding normalization and scale invariance
- Search latency compliance (< 15ms per query)
- Preservation of retrieval ranking interface

The reference implementation is not the only accepted solution.
