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


def test_list_runs_with_status_filter():
    r = client.get("/api/runs?status=succeeded")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_cancel_run():
    payload = {
        "input_text": "A quick cancel test for synthDataLab pipeline." * 100,
        "mock": True,
        "workers": 1,
        "agent": False,
    }
    create = client.post("/api/runs", json=payload)
    assert create.status_code == 202
    run_id = create.json()["run_id"]

    cancel = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel.status_code == 200
    cancel_data = cancel.json()
    assert cancel_data["run_id"] == run_id
    assert cancel_data["status"] in {"canceled", "succeeded", "failed"}

    run_detail = client.get(f"/api/runs/{run_id}")
    assert run_detail.status_code == 200
    assert run_detail.json()["status"] in {"canceled", "succeeded", "failed"}
