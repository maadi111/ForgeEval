"""Security and integrity utilities for the evaluation harness (Starter implementation)."""



class DatasetIntegrityVerifier:
    """Verifies that evaluation fixtures remain unchanged during evaluation."""

    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.initial_count = len(dataset.get("records", []))

    def compute_fingerprint(self) -> str:
        """Naive fingerprint: only counts records."""
        return f"count_{self.initial_count}"

    def verify_integrity(self) -> bool:
        """Naive check: passes as long as record count hasn't changed."""
        current_count = len(self.dataset.get("records", []))
        return current_count == self.initial_count
