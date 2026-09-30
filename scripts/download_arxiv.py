"""Download arXiv papers about RAG and LLMs into data/pdfs/ and save their metadata.

Usage (from the repo root):
    python scripts/download_arxiv.py --list-only   # only print what would be downloaded
    python scripts/download_arxiv.py               # download PDFs + write metadata.jsonl
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import arxiv

PDF_DIR = Path("data/pdfs")
METADATA_PATH = PDF_DIR / "metadata.jsonl"

# Landmark papers we always want (ids verified against the arXiv API).
PINNED_IDS = [
    "2005.11401",  # RAG (Lewis et al.)
    "2004.04906",  # Dense Passage Retrieval
    "2002.08909",  # REALM
    "2312.10997",  # RAG survey
    "2309.15217",  # Ragas
    "2004.12832",  # ColBERT
    "2402.03216",  # BGE M3-Embedding
    "2310.11511",  # Self-RAG
    "2212.10496",  # HyDE
    "2307.03172",  # Lost in the Middle
    "2401.15884",  # Corrective RAG
]

# Exact phrases ("...") avoid matching each word separately; the category filter
# (cs.CL = language, cs.IR = information retrieval) removes off-topic papers.
_CATS = "(cat:cs.CL OR cat:cs.IR)"
QUERIES = [
    f'all:"retrieval-augmented generation" AND {_CATS}',
    f'all:"dense retrieval" AND {_CATS}',
    f'all:"reranking" AND all:"retrieval" AND {_CATS}',
    f'ti:"hallucination" AND abs:"language models" AND {_CATS}',
    f'(all:"RAG evaluation" OR all:"evaluating retrieval-augmented") AND {_CATS}',
]
TARGET = 40
PAUSE_SECONDS = 3  # arXiv asks for a pause between requests
MAX_ATTEMPTS = 3
USER_AGENT = "PaperLens/0.1 (personal RAG learning project)"


def paper_id(result: arxiv.Result) -> str:
    """Return the arXiv id without the version suffix (2005.11401v4 -> 2005.11401)."""
    return re.sub(r"v\d+$", "", result.get_short_id())


def collect_papers(client: arxiv.Client) -> list[arxiv.Result]:
    """Return up to TARGET unique papers: the pinned ones plus a share of each query."""
    papers: dict[str, arxiv.Result] = {}

    for result in client.results(arxiv.Search(id_list=PINNED_IDS)):
        papers[paper_id(result)] = result
    print(f"pinned: {len(papers)} papers")

    remaining = TARGET - len(papers)
    quota = -(-remaining // len(QUERIES))  # ceiling division: papers per query

    for query in QUERIES:
        if len(papers) >= TARGET:
            break
        search = arxiv.Search(
            query=query,
            max_results=quota * 3,  # extra room: some results will be duplicates
            sort_by=arxiv.SortCriterion.Relevance,
        )
        taken = 0
        for result in client.results(search):
            pid = paper_id(result)
            if pid in papers:
                continue
            papers[pid] = result
            taken += 1
            if taken == quota or len(papers) >= TARGET:
                break
        print(f"{query!r}: {taken} new papers")

    return list(papers.values())


def download_pdf(result: arxiv.Result, path: Path) -> None:
    """Download to a temporary file first so an interrupted run leaves no broken PDF."""
    tmp_path = path.with_suffix(".part")
    if result.pdf_url is None:
        raise urllib.error.URLError("paper has no PDF url")
    request = urllib.request.Request(result.pdf_url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                tmp_path.write_bytes(response.read())
            tmp_path.replace(path)
            return
        except (urllib.error.URLError, TimeoutError):
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(PAUSE_SECONDS * attempt * 2)  # back off: 6s, 12s, ...


def to_metadata(result: arxiv.Result, path: Path) -> dict:
    """Fields that map directly to the `documents` table."""
    return {
        "arxiv_id": paper_id(result),
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "year": result.published.year,
        "source_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only", action="store_true", help="print papers, do not download")
    args = parser.parse_args()

    client = arxiv.Client(delay_seconds=PAUSE_SECONDS, num_retries=3)
    papers = collect_papers(client)
    print(f"\n{len(papers)} unique papers selected\n")

    if args.list_only:
        for result in papers:
            print(f"{paper_id(result):>12}  {result.published.year}  {result.title}")
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    failed: list[str] = []
    for result in papers:
        pid = paper_id(result)
        path = PDF_DIR / f"{pid.replace('/', '_')}.pdf"  # old-style ids contain '/'
        if path.exists():
            print(f"skip     {pid} (already downloaded)")
        else:
            print(f"download {pid}  {result.title}")
            try:
                download_pdf(result, path)
            except (urllib.error.URLError, TimeoutError) as error:
                print(f"FAILED   {pid}: {error}")
                failed.append(pid)
                continue  # no PDF, so no metadata entry; a re-run will retry it
            time.sleep(PAUSE_SECONDS)
        metadata.append(to_metadata(result, path))

    # Rewritten from scratch each run, so re-running never duplicates lines.
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(metadata)} entries to {METADATA_PATH}")
    if failed:
        print(f"{len(failed)} failed, run the script again to retry: {', '.join(failed)}")


if __name__ == "__main__":
    main()
