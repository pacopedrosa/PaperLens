"""Integration tests against the real PostgreSQL. Skipped when the database is down.

Everything runs inside an outer transaction that is always rolled back (see the
`conn` fixture), so the real corpus is never modified.
"""

import psycopg
import pytest
from pgvector.psycopg import register_vector

from paperlens.config import settings
from paperlens.ingest.store import document_exists, save_document
from paperlens.retrieval import search as search_module

DIMENSIONS = 1024


def unit_vector(index: int) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[index] = 1.0
    return vector


@pytest.fixture
def conn():
    try:
        connection = psycopg.connect(settings.database_url, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("database not available")
    register_vector(connection)
    # force_rollback opens an outer transaction that is always rolled back, so the
    # transactions inside save_document become savepoints instead of real commits.
    with connection.transaction(force_rollback=True):
        yield connection
    connection.close()


def metadata(arxiv_id: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Test paper {arxiv_id}",
        "authors": ["A. Author", "B. Author"],
        "year": 2024,
        "source_path": f"data/pdfs/{arxiv_id}.pdf",
    }


def test_save_document_stores_document_and_chunks(conn):
    chunks = [{"page": 1, "content": "first"}, {"page": 2, "content": "second"}]
    assert not document_exists(conn, "test.0001")

    save_document(conn, metadata("test.0001"), chunks, [unit_vector(0), unit_vector(1)])

    assert document_exists(conn, "test.0001")
    rows = conn.execute(
        "SELECT c.page, c.content FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.arxiv_id = 'test.0001' ORDER BY c.page"
    ).fetchall()
    assert rows == [(1, "first"), (2, "second")]


def test_nul_characters_are_removed_before_saving(conn):
    chunks = [{"page": 1, "content": "bad\x00text"}]

    save_document(conn, metadata("test.0002"), chunks, [unit_vector(0)])

    content = conn.execute(
        "SELECT c.content FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE d.arxiv_id = 'test.0002'"
    ).fetchone()[0]
    assert content == "badtext"


def test_a_second_document_with_the_same_arxiv_id_is_rejected(conn):
    chunk = [{"page": 1, "content": "x"}]
    save_document(conn, metadata("test.0003"), chunk, [unit_vector(0)])

    with pytest.raises(psycopg.errors.UniqueViolation):
        save_document(conn, metadata("test.0003"), chunk, [unit_vector(0)])


def test_search_returns_the_most_similar_chunk_first(conn, monkeypatch):
    chunks = [{"page": 1, "content": "about cats"}, {"page": 2, "content": "about dogs"}]
    save_document(conn, metadata("test.0004"), chunks, [unit_vector(5), unit_vector(6)])
    # The question is embedded as the same vector as the second chunk.
    monkeypatch.setattr(search_module, "embed_texts", lambda texts: [unit_vector(6)])

    results = search_module.search(conn, "anything", k=3)

    assert results[0]["arxiv_id"] == "test.0004"
    assert results[0]["content"] == "about dogs"
    assert results[0]["page"] == 2
    assert results[0]["similarity"] == pytest.approx(1.0)
