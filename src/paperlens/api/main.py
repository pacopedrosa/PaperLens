from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

from paperlens import rag
from paperlens.config import settings


def get_conn():
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        yield conn


# Alias for "a database connection obtained with get_conn", used as an endpoint parameter type.
Conn = Annotated[psycopg.Connection, Depends(get_conn)]


app = FastAPI(title="Paperlens")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20, description="number of chunks to retrieve")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest, conn: Conn):
    """Ask a question and get an answer with sources."""
    return rag.answer_question(conn, request.question, request.k)
