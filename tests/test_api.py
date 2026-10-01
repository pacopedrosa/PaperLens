import pytest
from fastapi.testclient import TestClient

from paperlens.api import main

ANSWER = {
    "answer": "Something [1].",
    "sources": [
        {
            "number": 1,
            "arxiv_id": "1111.0001",
            "title": "Paper A",
            "page": 2,
            "snippet": "text",
            "similarity": 0.9,
        }
    ],
}


@pytest.fixture
def client(monkeypatch):
    """API with the database and the RAG pipeline replaced by fakes."""
    asked = []

    def fake_answer(conn, question, k):
        asked.append((question, k))
        return ANSWER

    main.app.dependency_overrides[main.get_conn] = lambda: None
    monkeypatch.setattr(main.rag, "answer_question", fake_answer)
    test_client = TestClient(main.app)
    test_client.asked = asked
    yield test_client
    main.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_serves_the_chat_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PaperLens" in response.text


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.json() == ANSWER
    assert client.asked == [("What is RAG?", 5)]  # k defaults to 5


def test_k_is_passed_to_the_pipeline(client):
    client.post("/ask", json={"question": "What is RAG?", "k": 8})

    assert client.asked == [("What is RAG?", 8)]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "ab"},
        {"question": "x" * 1001},
        {"question": "valid question", "k": 0},
        {"question": "valid question", "k": 21},
        {},
    ],
)
def test_invalid_requests_are_rejected_without_calling_the_pipeline(client, payload):
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    assert client.asked == []
