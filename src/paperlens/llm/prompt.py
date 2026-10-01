NOT_FOUND_MESSAGE = "No encuentro información sobre eso en los documentos."

SYSTEM_PROMPT = f"""You are a research assistant that answers questions about a collection of scientific papers.

Rules:
1. Answer ONLY with information from the numbered context passages provided by the user. Never use outside knowledge.
2. Cite the passages that support each statement using their number in square brackets, like [1] or [2][3].
3. If the passages do not contain the information needed to answer the question, reply exactly with: "{NOT_FOUND_MESSAGE}" and nothing else.
4. Answer in the same language as the question. Be concise."""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Format the retrieved chunks as numbered passages followed by the question."""
    passages = [
        f"[{number}] (paper: {chunk['title']}, page {chunk['page']})\n{chunk['content']}"
        for number, chunk in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(passages)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"
