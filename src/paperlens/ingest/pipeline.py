"""Ingest every paper listed in data/pdfs/metadata.jsonl into PostgreSQL.

Usage (from the repo root):
    python -m paperlens.ingest.pipeline            # ingest everything not yet stored
    python -m paperlens.ingest.pipeline --limit 2  # only look at the first 2 papers
"""

import argparse
import json
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from paperlens.config import settings
from paperlens.ingest.chunking import chunk_pages
from paperlens.ingest.embeddings import embed_texts
from paperlens.ingest.parsing import extract_pages
from paperlens.ingest.store import document_exists, save_document

METADATA_PATH = Path("data/pdfs/metadata.jsonl")


def ingest_paper(conn: psycopg.Connection, metadata: dict) -> int:
    """Run the full pipeline for one paper and return how many chunks were stored."""
    pages = extract_pages(Path(metadata["source_path"]))
    chunks = chunk_pages(pages)
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    save_document(conn, metadata, chunks, vectors)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only process the first N papers")
    args = parser.parse_args()

    papers = [json.loads(line) for line in METADATA_PATH.read_text().splitlines()]
    papers = papers[: args.limit]

    # autocommit=True so that `conn.transaction()` in save_document is a real
    # transaction that commits when it ends, not a savepoint inside an open one.
    conn = psycopg.connect(settings.database_url, autocommit=True)
    register_vector(conn)

    stored = skipped = 0
    failed: list[str] = []
    for number, metadata in enumerate(papers, start=1):
        arxiv_id = metadata["arxiv_id"]
        label = f"[{number}/{len(papers)}] {arxiv_id}"

        if document_exists(conn, arxiv_id):
            print(f"{label} skip (already stored)")
            skipped += 1
            continue

        started = time.time()
        try:
            n_chunks = ingest_paper(conn, metadata)
        except Exception as error:  # noqa: BLE001 - one bad paper must not stop the run
            print(f"{label} FAILED: {error!r}")
            failed.append(arxiv_id)
            continue
        print(f"{label} stored {n_chunks} chunks in {time.time() - started:.0f}s")
        stored += 1

    print(f"\nstored: {stored}, skipped: {skipped}, failed: {len(failed)}")
    if failed:
        print("run again to retry:", ", ".join(failed))


if __name__ == "__main__":
    main()
