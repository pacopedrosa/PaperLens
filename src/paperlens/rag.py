import psycopg

from paperlens.llm.client import generate
from paperlens.llm.prompt import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from paperlens.retrieval.search import search

SNIPPET_CHARS = 300


def answer_question(conn: psycopg.Connection, question: str, k: int = 5) -> dict:
    """Retrieve the top-k chunks, ask the LLM and return the answer with its sources.

    Source numbers match the [n] citations the LLM was told to use.
    """
    chunks = search(conn, question, k=k)
    answer = generate(SYSTEM_PROMPT, build_user_prompt(question, chunks)).strip()

    # If the model says it found nothing, the retrieved chunks are not real sources.
    if answer == NOT_FOUND_MESSAGE:
        return {"answer": answer, "sources": []}

    sources = [
        {
            "number": number,
            "arxiv_id": chunk["arxiv_id"],
            "title": chunk["title"],
            "page": chunk["page"],
            "snippet": chunk["content"][:SNIPPET_CHARS],
            "similarity": chunk["similarity"],
        }
        for number, chunk in enumerate(chunks, start=1)
    ]
    return {"answer": answer, "sources": sources}
