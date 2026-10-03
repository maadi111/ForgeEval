"""Security and verification engine for solution-b using isolated state and SHA-256."""

import hashlib


class DatasetIntegrityVerifier:
    """Verifies that dataset integrity is preserved by comparing state digests."""

    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.reference_digest = self.compute_fingerprint()

    def compute_fingerprint(self) -> str:
        """Deterministic digest of all features and ground truth labels."""
        h = hashlib.sha256()
        records = self.dataset.get("records", [])
        for r in records:
            h.update(str(r.get("sample_id")).encode())
            feats = r.get("features", [])
            for val in feats:
                h.update(f"{float(val):.4f}".encode())
            h.update(str(r.get("label")).encode())
        return h.hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that current memory state matches original reference digest."""
        current_digest = self.compute_fingerprint()
        return current_digest == self.reference_digest
