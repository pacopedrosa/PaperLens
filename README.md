# PaperLens

RAG sobre papers de arXiv con citas verificables y evaluación automática.

## Objetivo
Preguntas en lenguaje natural sobre una colección de papers, con respuestas
basadas solo en los documentos y citas a paper y página.

## Stack
Python · FastAPI · PostgreSQL + pgvector · bge-m3 · Claude API · Next.js

## Puesta en marcha
    cp .env.example .env
    docker compose up -d
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Roadmap
- [ ] Etapa 1: ingesta + /ask básico
- [ ] Etapa 2: golden set + métricas
- [ ] Etapa 3: chunking, búsqueda híbrida, reranking
- [ ] Etapa 4: interfaz con streaming y citas
- [ ] Etapa 5: observabilidad, caché y CI con evaluación

## Resultados de evaluación
(Aquí irá una tabla con recall@k y MRR por cada cambio.)