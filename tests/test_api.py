"""Tests for standalone backend API (kaggle_chat.api)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from kaggle_chat.api import DISCLAIMER, EMERGENCY_PREFIX, app


@pytest.fixture
def client_mock(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MOCK", "1")
    return TestClient(app)


@pytest.fixture
def client_live(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("MOCK", raising=False)
    return TestClient(app)


def test_health_mock(client_mock: TestClient) -> None:
    resp = client_mock.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "mock"
    assert data["model"] == "medpsy-4b"
    assert data["model_loaded"] is False
    assert data["disclaimer"] == DISCLAIMER
    assert "X-Request-ID" in resp.headers


def test_emergency_path_skips_model(client_live: TestClient) -> None:
    # POST /api/v1/emergency
    resp_em = client_live.post("/api/v1/emergency", json={"query": "Patient has severe chest pain"})
    assert resp_em.status_code == 200
    data_em = resp_em.json()
    assert data_em["is_emergency"] is True
    assert data_em["reason"] == "Chest pain"
    assert data_em["emergency_checker"] == "red_flags"
    assert data_em["disclaimer"] == DISCLAIMER

    # POST /api/v1/chat in LIVE mode (where non-emergency raises 501)
    # The emergency check must fire first, returning 200 and skipping the model
    resp_chat = client_live.post(
        "/api/v1/chat",
        json={"session_id": "em-test-1", "query": "Patient has severe chest pain"},
    )
    assert resp_chat.status_code == 200
    data_chat = resp_chat.json()
    assert data_chat["red_flag"] is True
    assert data_chat["emergency_reason"] == "Chest pain"
    assert data_chat["emergency_checker"] == "red_flags"
    assert data_chat["abstained"] is False
    assert data_chat["citations"] == []
    assert data_chat["disclaimer"] == DISCLAIMER
    assert data_chat["mode"] == "live"
    assert EMERGENCY_PREFIX.strip() in data_chat["answer"]


@pytest.mark.parametrize(
    "query",
    [
        "My father has crushing chest pain and is sweating",
        "She cannot breathe",
        "baby 2 weeks old with fever",
        "pregnant woman with bleeding",
        "he had a seizure",
    ],
)
def test_emergency_red_flag_true(client_mock: TestClient, query: str) -> None:
    resp = client_mock.post("/api/v1/chat", json={"session_id": "test-em", "query": query})
    assert resp.status_code == 200
    data = resp.json()
    assert data["red_flag"] is True
    assert data["emergency_reason"] is not None
    assert data["emergency_checker"] == "red_flags"
    assert EMERGENCY_PREFIX.strip() in data["answer"]


@pytest.mark.parametrize(
    "query",
    [
        "no chest pain, only mild cough",
        "child has fever for 3 days",
    ],
)
def test_emergency_red_flag_false(client_mock: TestClient, query: str) -> None:
    resp = client_mock.post("/api/v1/chat", json={"session_id": "test-non-em", "query": query})
    assert resp.status_code == 200
    data = resp.json()
    assert data["red_flag"] is False
    assert data["emergency_reason"] is None
    assert data["emergency_checker"] == "red_flags"


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "my mother is having trouble breathing",
            marks=pytest.mark.xfail(
                reason="regex matches 'difficulty breathing' or 'cannot breathe', not 'trouble breathing'"
            ),
        ),
        pytest.param(
            "pain in the chest",
            marks=pytest.mark.xfail(
                reason="regex matches 'chest pain', not inverted 'pain in the chest'"
            ),
        ),
        pytest.param(
            "he is not waking up",
            marks=pytest.mark.xfail(
                reason="regex matches 'unconscious/unresponsive', not 'not waking up'"
            ),
        ),
        pytest.param(
            "bleeding will not stop",
            marks=pytest.mark.xfail(
                reason="regex matches 'heavy bleeding', not 'bleeding will not stop'"
            ),
        ),
    ],
)
def test_emergency_expected_misses(client_mock: TestClient, query: str) -> None:
    resp = client_mock.post("/api/v1/chat", json={"session_id": "test-xfail", "query": query})
    assert resp.status_code == 200
    assert resp.json()["red_flag"] is True


def test_mock_chat_answer_with_two_citations(client_mock: TestClient) -> None:
    resp = client_mock.post(
        "/api/v1/chat",
        json={"session_id": "test-sess-01", "query": "Child with fever and mild cough for two days."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-sess-01"
    assert data["red_flag"] is False
    assert data["abstained"] is False
    assert len(data["citations"]) == 2
    assert "id" in data["citations"][0] and "text" in data["citations"][0]
    assert "id" in data["citations"][1] and "text" in data["citations"][1]
    assert len(data["answer"]) > 0
    assert data["mode"] == "mock"
    assert data["latency_s"] >= 0.0
    assert data["disclaimer"] == DISCLAIMER


def test_abstain_case(client_mock: TestClient) -> None:
    resp = client_mock.post(
        "/api/v1/chat",
        json={"session_id": "test-sess-02", "query": "This query is completely unanswerable with the given clinical evidence."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["abstained"] is True
    assert data["red_flag"] is False
    assert data["citations"] == []
    assert "information" in data["answer"].lower() or "consult a doctor" in data["answer"].lower()
    assert data["disclaimer"] == DISCLAIMER


def test_disclaimer_present_across_endpoints(client_mock: TestClient) -> None:
    h_resp = client_mock.get("/api/v1/health")
    assert h_resp.json()["disclaimer"] == DISCLAIMER

    e_resp = client_mock.post("/api/v1/emergency", json={"query": "Routine checkup query."})
    assert e_resp.json()["disclaimer"] == DISCLAIMER

    c_resp = client_mock.post(
        "/api/v1/chat",
        json={"session_id": "disc-sess", "query": "Routine health query."},
    )
    assert c_resp.json()["disclaimer"] == DISCLAIMER


def test_empty_query_rejected_with_422(client_mock: TestClient) -> None:
    # Empty string in chat
    r1 = client_mock.post("/api/v1/chat", json={"session_id": "s1", "query": ""})
    assert r1.status_code == 422

    # Whitespace-only string in chat
    r2 = client_mock.post("/api/v1/chat", json={"session_id": "s2", "query": "    "})
    assert r2.status_code == 422

    # Empty string in emergency
    r3 = client_mock.post("/api/v1/emergency", json={"query": ""})
    assert r3.status_code == 422

    # Whitespace-only string in emergency
    r4 = client_mock.post("/api/v1/emergency", json={"query": "   "})
    assert r4.status_code == 422


def test_live_mode_returns_501(client_live: TestClient) -> None:
    resp = client_live.post(
        "/api/v1/chat",
        json={"session_id": "live-sess", "query": "What is the recommended treatment for hypertension?"},
    )
    assert resp.status_code == 501
    assert "live" in resp.json()["detail"].lower()
