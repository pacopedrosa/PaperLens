import pytest

from eval.run_eval import first_relevant_rank, load_golden_set, mrr, recall_at_k

QUESTION = {"arxiv_id": "A", "pages": [2, 3]}


def chunk(arxiv_id: str, page: int) -> dict:
    return {"arxiv_id": arxiv_id, "page": page}


# --- first_relevant_rank ---


def test_rank_is_one_based():
    assert first_relevant_rank([chunk("A", 3), chunk("B", 1)], QUESTION) == 1


def test_rank_skips_wrong_paper_and_wrong_page():
    # B has page 3 (wrong paper) and A page 9 (wrong page): only the third chunk is correct
    results = [chunk("B", 3), chunk("A", 9), chunk("A", 2)]
    assert first_relevant_rank(results, QUESTION) == 3


def test_rank_is_none_when_nothing_matches():
    assert first_relevant_rank([chunk("B", 3), chunk("A", 9)], QUESTION) is None


def test_rank_of_no_results_is_none():
    assert first_relevant_rank([], QUESTION) is None


# --- recall_at_k and mrr ---

RANKS = [1, 2, None, 5, 1]


@pytest.mark.parametrize(("k", "expected"), [(1, 0.4), (3, 0.6), (5, 0.8)])
def test_recall_at_k(k, expected):
    assert recall_at_k(RANKS, k) == pytest.approx(expected)


def test_mrr():
    assert mrr(RANKS) == pytest.approx(0.54)


def test_metrics_of_a_perfect_system():
    assert recall_at_k([1, 1, 1], 1) == 1.0
    assert mrr([1, 1, 1]) == 1.0


def test_metrics_of_a_system_that_finds_nothing():
    assert recall_at_k([None, None], 10) == 0.0
    assert mrr([None, None]) == 0.0


# --- the golden set itself ---

TYPES = {"factual", "conceptual", "paraphrased", "unanswerable"}


@pytest.fixture(scope="module")
def golden_set():
    return load_golden_set()


def test_golden_set_ids_are_unique(golden_set):
    ids = [q["id"] for q in golden_set]
    assert len(ids) == len(set(ids))


def test_golden_set_questions_are_well_formed(golden_set):
    for q in golden_set:
        assert q["type"] in TYPES, q["id"]
        assert 3 <= len(q["question"]) <= 1000, q["id"]  # same limits as the API
        assert q["answer"].strip(), q["id"]


def test_answerable_questions_point_to_a_paper_and_pages(golden_set):
    for q in golden_set:
        if q["type"] == "unanswerable":
            continue
        assert q["arxiv_id"], q["id"]
        assert q["pages"], q["id"]
        assert all(isinstance(page, int) and page >= 1 for page in q["pages"]), q["id"]


def test_unanswerable_questions_have_no_paper_or_pages(golden_set):
    unanswerable = [q for q in golden_set if q["type"] == "unanswerable"]
    assert unanswerable
    for q in unanswerable:
        assert q["arxiv_id"] is None and q["pages"] == [], q["id"]
