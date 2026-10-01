import psycopg
from pgvector import Vector


def document_exists(conn: psycopg.Connection, arxiv_id: str) -> bool:
    """True if this paper is already stored (used to make ingestion idempotent)."""
    row = conn.execute(
        "SELECT 1 FROM documents WHERE arxiv_id = %s", (arxiv_id,)
    ).fetchone()
    return row is not None


def save_document(
    conn: psycopg.Connection,
    metadata: dict,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    """Insert one document and all its chunks atomically.

    Either everything is saved or nothing is, so a crash never leaves a document
    without chunks that a later run would wrongly consider already ingested.
    """
    with conn.transaction():
        document_id = conn.execute(
            "INSERT INTO documents (arxiv_id, title, authors, year, source_path) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (
                metadata["arxiv_id"],
                metadata["title"],
                metadata["authors"],
                metadata["year"],
                metadata["source_path"],
            ),
        ).fetchone()[0]

        rows = [
            (
                document_id,
                chunk["page"],
                # PostgreSQL text columns cannot hold NUL characters, which some PDFs contain
                chunk["content"].replace("\x00", ""),
                Vector(vector),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, page, content, embedding) "
                "VALUES (%s, %s, %s, %s)",
                rows,
            )
