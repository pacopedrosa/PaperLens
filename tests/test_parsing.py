import pymupdf

from paperlens.ingest.parsing import extract_pages


def make_pdf(path, pages_text: list[str]) -> None:
    """Build a small PDF on the fly so the tests do not depend on real papers."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_page_numbers_start_at_one_and_follow_the_pdf(tmp_path):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, ["first page", "second page"])

    pages = extract_pages(pdf)

    assert [number for number, _ in pages] == [1, 2]
    assert "first page" in pages[0][1]
    assert "second page" in pages[1][1]


def test_pages_without_text_are_skipped_but_numbering_is_kept(tmp_path):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, ["first page", "", "third page"])

    pages = extract_pages(pdf)

    assert [number for number, _ in pages] == [1, 3]
