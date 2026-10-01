from pathlib import Path

import pymupdf


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append((index + 1, text))
    return pages
