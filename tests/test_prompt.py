from paperlens.llm.prompt import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt

CHUNKS = [
    {"title": "Paper A", "page": 2, "content": "alpha text"},
    {"title": "Paper B", "page": 7, "content": "beta text"},
]


def test_passages_are_numbered_from_one_with_paper_and_page():
    prompt = build_user_prompt("What is it?", CHUNKS)

    assert "[1] (paper: Paper A, page 2)\nalpha text" in prompt
    assert "[2] (paper: Paper B, page 7)\nbeta text" in prompt


def test_question_comes_after_the_context():
    prompt = build_user_prompt("What is it?", CHUNKS)

    assert prompt.index("beta text") < prompt.index("Question: What is it?")


def test_system_prompt_forces_the_not_found_answer_and_citations():
    assert NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT
