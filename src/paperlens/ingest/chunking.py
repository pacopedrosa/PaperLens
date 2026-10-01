CHUNK_WORDS = 300
OVERLAP_WORDS = 40


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for page_number, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + CHUNK_WORDS]
            chunks.append({"page": page_number, "content": " ".join(chunk_words)})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks
