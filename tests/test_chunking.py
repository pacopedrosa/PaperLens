from paperlens.ingest.chunking import CHUNK_WORDS, OVERLAP_WORDS, chunk_pages


def make_words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def test_empty_input_gives_no_chunks():
    assert chunk_pages([]) == []


def test_short_page_gives_a_single_chunk_with_its_page():
    chunks = chunk_pages([(4, make_words(50))])

    assert len(chunks) == 1
    assert chunks[0]["page"] == 4


def test_page_of_exactly_chunk_size_is_not_split():
    assert len(chunk_pages([(1, make_words(CHUNK_WORDS))])) == 1


def test_long_page_is_split_with_overlap_between_consecutive_chunks():
    chunks = chunk_pages([(1, make_words(700))])

    first, second = (chunk["content"].split() for chunk in chunks[:2])
    assert len(chunks) > 1
    assert len(first) == CHUNK_WORDS
    assert first[-OVERLAP_WORDS:] == second[:OVERLAP_WORDS]


def test_no_words_are_lost():
    chunks = chunk_pages([(1, make_words(700))])

    assert chunks[0]["content"].split()[0] == "w0"
    assert chunks[-1]["content"].split()[-1] == "w699"


def test_chunks_never_mix_pages():
    chunks = chunk_pages([(1, make_words(400, "a")), (2, make_words(400, "b"))])

    for chunk in chunks:
        expected = "a" if chunk["page"] == 1 else "b"
        assert {word[0] for word in chunk["content"].split()} == {expected}
