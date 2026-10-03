"""Tests for the Submissions API routes."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_submission_lifecycle():
    """Verify POST /submissions returns 202 and GET /submissions/{id} retrieves it."""
    payload = {
        "task_id": "feature-leakage-v1",
        "image": "benchmarks/feature-leakage/workspace",
    }
    create_resp = client.post("/submissions", json=payload)
    assert create_resp.status_code == 202
    created_data = create_resp.json()
    assert "submission_id" in created_data
    assert created_data["task_id"] == "feature-leakage-v1"
    assert created_data["status"] in ["pending", "running", "done", "failed"]

    sub_id = created_data["submission_id"]
    get_resp = client.get(f"/submissions/{sub_id}")
    assert get_resp.status_code == 200
    retrieved_data = get_resp.json()
    assert retrieved_data["submission_id"] == sub_id


def test_submission_not_found():
    """GET /submissions/{id} should return 404 for unknown submission."""
    response = client.get("/submissions/non-existent-sub")
    assert response.status_code == 404
