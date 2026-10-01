import psycopg
from pgvector import Vector

from paperlens.ingest.embeddings import embed_texts


def search(conn: psycopg.Connection, question: str, k: int = 5) -> list[dict]:
    query_vector = Vector(embed_texts([question])[0])
    rows = conn.execute(
        """
        SELECT d.arxiv_id, d.title, c.page, c.content,
               1 - (c.embedding <=> %s) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> %s
        LIMIT %s
        """,
        (query_vector, query_vector, k),
    ).fetchall()
    return[
        {
            "arxiv_id":arxiv_id, 
            "title":title,
            "page":page,
            "content":content,
            "similarity":float(similarity)
        }
        for arxiv_id, title, page, content, similarity in rows
    ]
    