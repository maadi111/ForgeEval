"""Tests for the Task Registry API routes."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_tasks():
    """GET /tasks should return all discovered tasks."""
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert "total" in data
    assert data["total"] >= 1

    task_ids = [t["id"] for t in data["tasks"]]
    assert "feature-leakage-v1" in task_ids
    assert "model-serving-v1" in task_ids
    assert "retrieval-drift-v1" in task_ids
    assert "adversarial-grading-v1" in task_ids
    assert "gpu-optimization-v1" in task_ids


def test_get_task_success():
    """GET /tasks/{task_id} should return details of a known task."""
    response = client.get("/tasks/feature-leakage-v1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "feature-leakage-v1"
    assert data["title"] == "Point-in-Time Feature Leakage"
    assert data["difficulty"] == "medium"
    assert data["category"] == "production-ml"

    response_b3 = client.get("/tasks/model-serving-v1")
    assert response_b3.status_code == 200
    data_b3 = response_b3.json()
    assert data_b3["id"] == "model-serving-v1"
    assert data_b3["difficulty"] == "hard"
    assert data_b3["category"] == "model-serving"

    response_b2 = client.get("/tasks/retrieval-drift-v1")
    assert response_b2.status_code == 200
    data_b2 = response_b2.json()
    assert data_b2["id"] == "retrieval-drift-v1"
    assert data_b2["difficulty"] == "medium"
    assert data_b2["category"] == "information-retrieval"

    response_b5 = client.get("/tasks/adversarial-grading-v1")
    assert response_b5.status_code == 200
    data_b5 = response_b5.json()
    assert data_b5["id"] == "adversarial-grading-v1"
    assert data_b5["difficulty"] == "Hard"

    response_b4 = client.get("/tasks/gpu-optimization-v1")
    assert response_b4.status_code == 200
    data_b4 = response_b4.json()
    assert data_b4["id"] == "gpu-optimization-v1"
    assert data_b4["difficulty"] == "Hard"


def test_get_task_not_found():
    """GET /tasks/{task_id} should return 404 for an unknown task."""
    response = client.get("/tasks/non-existent-task")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
