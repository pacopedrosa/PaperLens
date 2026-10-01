import pytest

from paperlens import rag
from paperlens.llm.prompt import NOT_FOUND_MESSAGE

CHUNKS = [
    {"arxiv_id": "1111.0001", "title": "Paper A", "page": 2, "content": "a" * 500, "similarity": 0.9},
    {"arxiv_id": "2222.0002", "title": "Paper B", "page": 5, "content": "short", "similarity": 0.7},
]


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace retrieval and the LLM, and record the prompt the LLM receives."""
    calls = {}

    def install(answer: str):
        monkeypatch.setattr(rag, "search", lambda conn, question, k=5: CHUNKS)

        def fake_generate(system: str, user: str) -> str:
            calls["system"], calls["user"] = system, user
            return answer

        monkeypatch.setattr(rag, "generate", fake_generate)
        return calls

    return install


def test_sources_are_numbered_like_the_citations(fake_llm):
    fake_llm("It works like this [1].")

    result = rag.answer_question(None, "How?")

    assert result["answer"] == "It works like this [1]."
    assert [s["number"] for s in result["sources"]] == [1, 2]
    assert [s["arxiv_id"] for s in result["sources"]] == ["1111.0001", "2222.0002"]
    assert result["sources"][0]["page"] == 2


def test_snippets_are_truncated(fake_llm):
    fake_llm("answer [1]")

    sources = rag.answer_question(None, "How?")["sources"]

    assert len(sources[0]["snippet"]) == rag.SNIPPET_CHARS
    assert sources[1]["snippet"] == "short"


def test_not_found_answer_returns_no_sources(fake_llm):
    fake_llm(f"  {NOT_FOUND_MESSAGE}\n")

    result = rag.answer_question(None, "Capital of Mongolia?")

    assert result == {"answer": NOT_FOUND_MESSAGE, "sources": []}


def test_llm_receives_the_retrieved_chunks_and_the_question(fake_llm):
    calls = fake_llm("ok")

    rag.answer_question(None, "My question")

    assert "Paper A" in calls["user"] and "Paper B" in calls["user"]
    assert "Question: My question" in calls["user"]
    assert NOT_FOUND_MESSAGE in calls["system"]
