# PaperLens

Ask questions in natural language about a collection of arXiv papers and get answers **grounded only in those documents**, with citations to the paper and page. If the answer is not in the corpus, PaperLens says so instead of making something up.

<!-- TODO: add a real screenshot at docs/screenshot.png and reference it here -->

## Features

- **Grounded answers with citations**: every claim is cited as `[1]`, `[2]`, and each citation links to the source paper at the exact page.
- **Honest refusals**: when the retrieved context does not contain the answer, the system replies *"No encuentro información sobre eso en los documentos."* and returns no sources.
- **Fully local and free**: embeddings (`BAAI/bge-m3`) and the LLM (Ollama) run on your machine. No paid API is needed.
- **Provider-agnostic LLM layer**: any OpenAI-compatible endpoint works (Ollama, Groq, Gemini, OpenRouter). Switching provider means changing three variables in `.env`, no code.
- **Idempotent ingestion**: re-running the pipeline skips papers that are already stored.
- **Web chat** with Markdown and LaTeX rendering, clickable citations, dark/light theme and a responsive layout.

## How it works

```
 INGESTION (once)                              QUESTION ANSWERING (per request)

 PDFs ──► extract text per page (PyMuPDF)      question ──► embed (bge-m3)
      ──► chunk: 300 words, 40 overlap                 ──► top-k cosine search (pgvector, HNSW)
      ──► embed chunks (bge-m3, batches)               ──► prompt: rules + numbered passages + question
      ──► store in PostgreSQL + pgvector               ──► LLM (Ollama, via OpenAI-compatible API)
                                                       ──► answer + sources (paper, page, snippet)
```

The whole pipeline is written by hand, with no LangChain or LlamaIndex, to make every step explicit.

## Stack

Python 3.11+ · FastAPI · PostgreSQL 16 + pgvector · sentence-transformers (`BAAI/bge-m3`, 1024 dimensions) · PyMuPDF · Ollama (`qwen2.5:7b` by default) · `openai` SDK as a plain HTTP client · pydantic-settings · pytest · ruff

## Getting started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose
- [uv](https://docs.astral.sh/uv/) (or any way to get Python 3.11+)
- [Ollama](https://ollama.com/download)

### Setup

```bash
git clone <this-repo> && cd PaperLens

cp .env.example .env              # adjust values if needed
docker compose up -d              # PostgreSQL + pgvector (creates the schema on first start)

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

ollama pull qwen2.5:7b            # about 4.7 GB, one time
```

### Build the corpus and run

```bash
python scripts/download_arxiv.py          # downloads ~40 papers into data/pdfs/ + metadata.jsonl
python -m paperlens.ingest.pipeline       # parse, chunk, embed and store (~20 min on CPU, once)
uvicorn paperlens.api.main:app --reload   # http://localhost:8000
```

Open <http://localhost:8000> for the chat, or <http://localhost:8000/docs> for the interactive API docs.

### Ask from the command line

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Ragas measure?", "k": 5}'
```

```json
{
  "answer": "Ragas measures faithfulness, answer relevance and context relevance [2].",
  "sources": [
    {"number": 2, "arxiv_id": "2309.15217", "title": "Ragas: ...", "page": 5,
     "snippet": "an evaluation framework that can assess faithfulness ...", "similarity": 0.55}
  ]
}
```

Each source `number` matches the `[n]` citations in the answer. On a CPU-only machine an answer takes about a minute, almost all of it spent in the LLM.

## Configuration

All settings are environment variables, read from `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | PostgreSQL connection string. Must match the `POSTGRES_*` values |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_PORT` | `paperlens` / `paperlens` / `paperlens` / `5432` | Used by `docker-compose.yml` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Changing it requires changing `vector(1024)` in `db/init.sql` |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | `ollama` | Ollama ignores it, other providers need a real key |
| `LLM_MODEL` | `qwen2.5:7b` | Model name as the provider knows it |

## Project structure

```
db/init.sql                  schema: documents, chunks (vector + full-text columns, HNSW and GIN indexes)
scripts/download_arxiv.py    download papers and their metadata from arXiv
src/paperlens/
  config.py                  settings (pydantic-settings)
  ingest/                    parsing.py · chunking.py · embeddings.py · store.py · pipeline.py
  retrieval/search.py        question -> top-k chunks by cosine similarity
  llm/client.py              generate(system, user): the only function that talks to the LLM
  llm/prompt.py              system prompt and context formatting
  rag.py                     retrieve -> prompt -> generate -> answer with sources
  api/                       FastAPI app and the web chat (static/index.html)
tests/                       pytest suite
eval/                        golden set and evaluation script (Stage 2)
```

## Design decisions

> Draft: to be rewritten in my own words.

- **Chunks never cross page boundaries.** Each chunk belongs to exactly one page, so a citation always points to the right page. The cost is that a sentence spanning two pages is cut.
- **Word-based chunk size.** 300 words is about 400 tokens. It avoids a tokenizer dependency while learning the mechanics; swapping in the real tokenizer is a one-function change.
- **Default PDF reading order.** `sort=True` in PyMuPDF interleaves the two columns of scientific papers and produces unreadable text, so the default order is used.
- **Cosine similarity with an HNSW index.** bge-m3 embeddings are normalized, and the index operator class matches the `<=>` operator used in the query.
- **Idempotency by `arxiv_id`.** The pipeline checks the database before computing embeddings, which is the expensive step. Each paper is saved in one transaction, so a crash never leaves a document without chunks.
- **No sources on a refusal.** When the model answers "not found", the retrieved chunks are not real sources, so none are returned.
- **Single LLM entry point.** Everything calls `generate(system, user)`, so changing provider never touches the rest of the code.

## Tests

```bash
python -m pytest -q
```

29 tests: unit tests for parsing, chunking and the prompt; tests for the RAG pipeline and the API with the LLM and the database replaced by fakes; and integration tests against the real PostgreSQL, which run inside a transaction that is always rolled back and are skipped if the database is not available.

## Roadmap

- [x] **Stage 1**: ingestion pipeline and basic `/ask`, without frameworks
- [ ] **Stage 2**: golden set (40-60 questions) and evaluation with recall@k and MRR
- [ ] **Stage 3**: better retrieval, each change measured: chunking strategies, hybrid search (BM25 + vectors), reranking
- [ ] **Stage 4**: Next.js + TypeScript interface with streaming and citations
- [ ] **Stage 5**: observability, caching and CI with automatic evaluation

## Evaluation results

*The table with recall@k and MRR for each change will go here (Stage 2 onwards).*

## License

MIT, see [LICENSE](LICENSE).
