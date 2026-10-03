"""Tests for Dashboard route and static asset delivery."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_dashboard_endpoint():
    """Verify that /dashboard/ returns the dashboard HTML page."""
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "ForgeEval" in response.text
    assert "Benchmarks &amp; Grading" in response.text or "Benchmarks & Grading" in response.text


def test_dashboard_data_js():
    """Verify that data.js is delivered."""
    response = client.get("/dashboard/data.js")
    assert response.status_code == 200
    assert "FORGEEVAL_DATA" in response.text
