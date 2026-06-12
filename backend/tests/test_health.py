"""Smoke tests for the API skeleton — validate wiring and the schema contract."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # The strict schema (extra="forbid") must expose exactly these keys.
    assert set(body) == {"status", "version"}


def test_model_info_is_stub(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["version"] is None


def test_root_ok(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
