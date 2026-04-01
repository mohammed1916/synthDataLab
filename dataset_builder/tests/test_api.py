"""Tests for the API layer in dataset_builder.api."""
import time
from fastapi.testclient import TestClient

from dataset_builder.api.app import app


client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_run_with_input_text_and_poll():
    payload = {
        "input_text": "This is a simple test document for synthDataLab.",
        "mock": True,
        "workers": 1,
        "agent": False,
    }
    r = client.post("/api/runs", json=payload)
    assert r.status_code == 202
    data = r.json()
    assert "run_id" in data
    run_id = data["run_id"]

    # Poll status until completion or timeout
    end_time = time.time() + 120
    status = None
    while time.time() < end_time:
        r2 = client.get(f"/api/runs/{run_id}")
        assert r2.status_code == 200
        details = r2.json()
        status = details.get("status")
        if status in {"succeeded", "failed"}:
            break
        time.sleep(1)

    assert status in {"succeeded", "failed"}
