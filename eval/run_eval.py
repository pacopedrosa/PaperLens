import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from paperlens.config import settings
from paperlens.retrieval.search import search

GOLDEN_SET = Path("eval/golden_set.jsonl")


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    """Read golden set object per line"""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def first_relevant_rank(results: list[dict], question: dict) -> int | None:
    """1-based position of the first correct chunk, or None if there is none."""
    position = 0
    for chunk in results:
        position = position + 1

        if chunk["arxiv_id"] != question["arxiv_id"]:
            continue  # wrong paper, look at the next chunk

        if chunk["page"] in question["pages"]:
            return position  # right paper and right page

    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose first correct chunk is within the top k."""
    hits = 0
    for rank in ranks:
        if rank is not None and rank <= k:
            hits = hits + 1
    return hits / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank: average of 1/rank, counting 0 when nothing was found."""
    total = 0.0
    for rank in ranks:
        if rank is not None:
            total = total + 1 / rank
    return total / len(ranks)


def main() -> None:
    questions = load_golden_set()
    ranks = []

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        for question in questions:
            results = search(conn, question["question"], k=10)
            rank = first_relevant_rank(results, question)
            ranks.append(rank)
            print(f"{question['id']}  rank={rank}  {question['question'][:60]}")

    print()
    print(f"questions: {len(questions)}")
    for k in (1, 3, 5, 10):
        print(f"recall@{k}: {recall_at_k(ranks, k):.2f}")
    print(f"MRR:      {mrr(ranks):.2f}")


if __name__ == "__main__":
    main()
