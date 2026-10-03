"""Cryptographic security and integrity verifier for solution-a."""

import hashlib


class DatasetIntegrityVerifier:
    """Verifies that evaluation fixtures remain strictly unaltered using SHA-256."""

    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.initial_hash = self.compute_fingerprint()

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA-256 hash across all sample records."""
        hasher = hashlib.sha256()
        records = self.dataset.get("records", [])
        for r in records:
            token = f"{r.get('sample_id')}|{r.get('features')}|{r.get('label')}"
            hasher.update(token.encode("utf-8"))
        return hasher.hexdigest()

    def verify_integrity(self) -> bool:
        """Recompute fingerprint and ensure zero state tampering occurred."""
        current_hash = self.compute_fingerprint()
        return current_hash == self.initial_hash
