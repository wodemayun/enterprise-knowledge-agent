# Enterprise Knowledge Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python service that ingests tenant-scoped documents, answers questions through a LangGraph RAG workflow with citations, and creates PostgreSQL tickets when evidence is insufficient.

**Architecture:** FastAPI exposes document, question, ticket, and health endpoints. Focused service interfaces isolate parsing, embeddings, Qdrant retrieval, LLM calls, and SQL persistence so tests can use deterministic local substitutes. LangGraph owns the explicit classify → retrieve → grade → answer/escalate → record workflow.

**Tech Stack:** Python 3.11, FastAPI, Pydantic Settings, LangGraph, Qdrant Client + FastEmbed, OpenAI Python SDK, SQLAlchemy 2.x, PostgreSQL, pytest, Docker Compose.

---

## File map

Create the project under `projects/enterprise-knowledge-agent/`.

- `pyproject.toml`: runtime and development dependencies plus pytest/ruff configuration.
- `.env.example`: documented environment variables without secrets.
- `.gitignore`: Python, test, environment, and local database artifacts.
- `app/main.py`: FastAPI application factory and exception handlers.
- `app/core/config.py`: validated environment configuration.
- `app/core/errors.py`: domain exceptions shared by API and services.
- `app/core/logging.py`: request-safe structured logging setup.
- `app/documents/models.py`: parsed document and chunk value objects.
- `app/documents/chunker.py`: deterministic overlapping text chunking.
- `app/documents/parser.py`: TXT, Markdown, and text-PDF extraction.
- `app/documents/service.py`: parse, chunk, embed, and index orchestration.
- `app/retrieval/models.py`: retrieved chunk model.
- `app/retrieval/embeddings.py`: embedding protocol, FastEmbed implementation, deterministic test implementation.
- `app/retrieval/vector_store.py`: vector-store protocol, Qdrant implementation, in-memory test implementation.
- `app/llm/provider.py`: LLM protocol, OpenAI Responses implementation, deterministic test implementation.
- `app/tickets/models.py`: SQLAlchemy ticket model and response value object.
- `app/tickets/repository.py`: ticket persistence interface and SQLAlchemy implementation.
- `app/runs/models.py`: SQLAlchemy Agent run model.
- `app/runs/repository.py`: Agent run persistence interface and SQLAlchemy implementation.
- `app/db.py`: engine, session factory, and table creation.
- `app/agent/state.py`: typed LangGraph state.
- `app/agent/workflow.py`: nodes, routing, graph compilation, and orchestration.
- `app/api/dependencies.py`: application dependency construction.
- `app/api/schemas.py`: request and response schemas.
- `app/api/routes.py`: health, upload, question, and ticket endpoints.
- `tests/`: behavior-focused unit and API integration tests.
- `sample_data/employee_handbook.md`: public demonstration data.
- `Dockerfile`, `compose.yaml`: reproducible local deployment.
- `README.md`: setup, operation, and architecture overview.
- `docs/learning-guide.md`: C#-to-Python guided walkthrough.
- `docs/interview-guide.md`: project pitch and technical follow-ups.
- `docs/architecture.md`: data flow and workflow diagrams.

## Task 1: Project scaffold and validated settings

**Files:**
- Create: `projects/enterprise-knowledge-agent/pyproject.toml`
- Create: `projects/enterprise-knowledge-agent/.gitignore`
- Create: `projects/enterprise-knowledge-agent/.env.example`
- Create: `projects/enterprise-knowledge-agent/app/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/core/__init__.py`
- Create: `projects/enterprise-knowledge-agent/tests/test_config.py`
- Create: `projects/enterprise-knowledge-agent/app/core/config.py`

- [ ] **Step 1: Add packaging and test configuration**

Create `pyproject.toml` with Python `>=3.11,<3.14`, the runtime dependencies listed in the header, and development dependencies `pytest`, `pytest-asyncio`, `httpx`, and `ruff`. Configure pytest with `pythonpath = ["."]` and `testpaths = ["tests"]`.

- [ ] **Step 2: Write the failing settings test**

```python
from app.core.config import Settings


def test_settings_use_safe_local_defaults():
    settings = Settings(_env_file=None)

    assert settings.app_name == "Enterprise Knowledge Agent"
    assert settings.llm_provider == "deterministic"
    assert settings.evidence_threshold == 0.72
    assert settings.retrieval_limit == 4
```

- [ ] **Step 3: Run the test and verify RED**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `app.core.config` does not exist.

- [ ] **Step 4: Implement minimal validated settings**

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Enterprise Knowledge Agent"
    environment: str = "development"
    database_url: str = "sqlite:///./knowledge_agent.db"
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "knowledge_chunks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "deterministic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.5"
    evidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    retrieval_limit: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    max_upload_bytes: int = 5 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/test_config.py -v`

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add projects/enterprise-knowledge-agent
git commit -m "build: scaffold enterprise knowledge agent"
```

## Task 2: Deterministic document chunking

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/documents/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/documents/models.py`
- Create: `projects/enterprise-knowledge-agent/app/documents/chunker.py`
- Create: `projects/enterprise-knowledge-agent/tests/documents/test_chunker.py`

- [ ] **Step 1: Write failing chunking tests**

```python
from app.documents.chunker import TextChunker


def test_chunker_preserves_overlap():
    chunker = TextChunker(chunk_size=10, overlap=3)

    chunks = chunker.split("abcdefghijklmnopqrstuvwxyz")

    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]


def test_chunker_rejects_empty_text():
    chunker = TextChunker(chunk_size=10, overlap=3)

    try:
        chunker.split("   \n")
    except ValueError as error:
        assert str(error) == "document contains no text"
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/documents/test_chunker.py -v`

Expected: FAIL because the document modules do not exist.

- [ ] **Step 3: Implement chunk model and chunker**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str
```

```python
import re

from app.documents.models import TextChunk


class TextChunker:
    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap smaller")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[TextChunk]:
        normalized = re.sub(r"[ \t]+", " ", text).strip()
        if not normalized:
            raise ValueError("document contains no text")
        step = self.chunk_size - self.overlap
        return [
            TextChunk(index=index, text=normalized[start : start + self.chunk_size])
            for index, start in enumerate(range(0, len(normalized), step))
        ]
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/documents/test_chunker.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/documents projects/enterprise-knowledge-agent/tests/documents
git commit -m "feat: add deterministic text chunking"
```

## Task 3: Safe TXT, Markdown, and PDF parsing

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/core/errors.py`
- Create: `projects/enterprise-knowledge-agent/app/documents/parser.py`
- Create: `projects/enterprise-knowledge-agent/tests/documents/test_parser.py`

- [ ] **Step 1: Write failing parser tests**

```python
import io

import pytest

from app.core.errors import UnsupportedDocumentError
from app.documents.parser import DocumentParser


def test_parser_reads_utf8_text():
    parser = DocumentParser()
    assert parser.parse("handbook.md", "年假制度".encode()) == "年假制度"


def test_parser_rejects_unknown_extension():
    parser = DocumentParser()
    with pytest.raises(UnsupportedDocumentError, match="unsupported document type"):
        parser.parse("handbook.docx", io.BytesIO(b"content").read())
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/documents/test_parser.py -v`

Expected: FAIL because parser and domain error modules do not exist.

- [ ] **Step 3: Implement parser and explicit domain error**

```python
class DomainError(Exception):
    pass


class UnsupportedDocumentError(DomainError):
    pass


class EmptyDocumentError(DomainError):
    pass


class ConfigurationError(DomainError):
    pass
```

```python
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.core.errors import EmptyDocumentError, UnsupportedDocumentError


class DocumentParser:
    def parse(self, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md"}:
            text = content.decode("utf-8-sig")
        elif suffix == ".pdf":
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            raise UnsupportedDocumentError("unsupported document type")
        if not text.strip():
            raise EmptyDocumentError("document contains no extractable text")
        return text
```

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/documents/test_parser.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/core/errors.py projects/enterprise-knowledge-agent/app/documents/parser.py projects/enterprise-knowledge-agent/tests/documents/test_parser.py
git commit -m "feat: parse supported knowledge documents"
```

## Task 4: Tenant-safe vector indexing and retrieval

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/retrieval/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/retrieval/models.py`
- Create: `projects/enterprise-knowledge-agent/app/retrieval/embeddings.py`
- Create: `projects/enterprise-knowledge-agent/app/retrieval/vector_store.py`
- Create: `projects/enterprise-knowledge-agent/tests/retrieval/test_vector_store.py`

- [ ] **Step 1: Write failing tenant-isolation test**

```python
from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.vector_store import InMemoryVectorStore, VectorDocument


def test_search_never_returns_another_tenants_chunks():
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(dimensions=16))
    store.add([
        VectorDocument("a-1", "tenant-a", "a.md", 0, "年假为五天"),
        VectorDocument("b-1", "tenant-b", "b.md", 0, "年假为十五天"),
    ])

    results = store.search("tenant-a", "年假多少天", limit=5)

    assert results
    assert {result.tenant_id for result in results} == {"tenant-a"}
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/retrieval/test_vector_store.py -v`

Expected: FAIL because retrieval modules do not exist.

- [ ] **Step 3: Implement protocols and deterministic in-memory behavior**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    tenant_id: str
    document_id: str
    source_name: str
    chunk_index: int
    text: str
    score: float
```

```python
import hashlib
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicEmbeddingProvider:
    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            normalized = "".join(text.lower().split())
            tokens = list(normalized) + [
                normalized[index : index + 2]
                for index in range(max(0, len(normalized) - 1))
            ]
            for token in tokens:
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors
```

Implement `VectorDocument`, a `VectorStore` protocol, and `InMemoryVectorStore` with cosine similarity. Filter candidates by exact `tenant_id` before sorting and limit results.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/retrieval/test_vector_store.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Add Qdrant implementation behind the same protocol**

Implement `QdrantVectorStore` using `QdrantClient`, cosine distance, UUID point IDs, payload metadata, and a mandatory Qdrant filter matching `tenant_id`. Add a local-mode test with deterministic vectors and assert the same tenant-isolation behavior.

- [ ] **Step 6: Verify all retrieval tests**

Run: `python -m pytest tests/retrieval -v`

Expected: all retrieval tests pass without network access.

- [ ] **Step 7: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/retrieval projects/enterprise-knowledge-agent/tests/retrieval
git commit -m "feat: add tenant-safe vector retrieval"
```

## Task 5: Document ingestion service

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/documents/service.py`
- Create: `projects/enterprise-knowledge-agent/tests/documents/test_service.py`

- [ ] **Step 1: Write failing ingestion test**

```python
from app.documents.chunker import TextChunker
from app.documents.parser import DocumentParser
from app.documents.service import DocumentService
from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.vector_store import InMemoryVectorStore


def test_ingestion_indexes_all_chunks_with_tenant_and_source():
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(16))
    service = DocumentService(DocumentParser(), TextChunker(10, 2), store)

    result = service.ingest("tenant-a", "policy.md", b"abcdefghijklmnopqrstuvwxyz")

    assert result.source_name == "policy.md"
    assert result.chunk_count == 4
    found = store.search("tenant-a", "abcdefghij", limit=10)
    assert found
    assert all(item.source_name == "policy.md" for item in found)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/documents/test_service.py -v`

Expected: FAIL because `DocumentService` does not exist.

- [ ] **Step 3: Implement minimal orchestration**

Define an immutable `IngestionResult(document_id, source_name, chunk_count)`. Generate one UUID document ID, convert each chunk to `VectorDocument`, add all chunks through `VectorStore`, and return the result.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/documents/test_service.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/documents projects/enterprise-knowledge-agent/tests/documents
git commit -m "feat: orchestrate document ingestion"
```

## Task 6: Ticket and Agent-run persistence

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/db.py`
- Create: `projects/enterprise-knowledge-agent/app/tickets/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/tickets/models.py`
- Create: `projects/enterprise-knowledge-agent/app/tickets/repository.py`
- Create: `projects/enterprise-knowledge-agent/app/runs/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/runs/models.py`
- Create: `projects/enterprise-knowledge-agent/app/runs/repository.py`
- Create: `projects/enterprise-knowledge-agent/tests/persistence/test_repositories.py`

- [ ] **Step 1: Write failing repository test using SQLite**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import Base
from app.runs.repository import SqlAlchemyRunRepository
from app.tickets.repository import SqlAlchemyTicketRepository


def test_ticket_and_run_are_persisted():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        tickets = SqlAlchemyTicketRepository(session)
        runs = SqlAlchemyRunRepository(session)
        ticket = tickets.create("tenant-a", "没有答案的问题", "insufficient_evidence")
        run = runs.create("tenant-a", "没有答案的问题", "escalated", 0.2, ticket.id, 18)
        session.commit()

        assert tickets.get(ticket.id).status == "open"
        assert run.ticket_id == ticket.id
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/persistence/test_repositories.py -v`

Expected: FAIL because database and repository modules do not exist.

- [ ] **Step 3: Implement SQLAlchemy models and repositories**

Use `DeclarativeBase`, timezone-aware UTC timestamps, integer primary keys, indexed `tenant_id`, and explicit repository methods. Repository methods call `session.flush()` so generated IDs are available while transaction ownership remains with the service/API boundary.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/persistence/test_repositories.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/db.py projects/enterprise-knowledge-agent/app/tickets projects/enterprise-knowledge-agent/app/runs projects/enterprise-knowledge-agent/tests/persistence
git commit -m "feat: persist tickets and agent runs"
```

## Task 7: Swappable LLM providers

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/llm/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/llm/provider.py`
- Create: `projects/enterprise-knowledge-agent/tests/llm/test_provider.py`

- [ ] **Step 1: Write failing provider tests**

```python
import pytest

from app.core.errors import ConfigurationError
from app.llm.provider import DeterministicProvider, OpenAIProvider
from app.retrieval.models import RetrievedChunk


def test_deterministic_provider_answers_from_context():
    provider = DeterministicProvider()
    chunk = RetrievedChunk("t", "d", "policy.md", 0, "年假为五天", 0.9)

    answer = provider.answer("年假多少天？", [chunk])

    assert "年假为五天" in answer


def test_openai_provider_requires_key():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIProvider(api_key=None, model="gpt-5.5")
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/llm/test_provider.py -v`

Expected: FAIL because provider module does not exist.

- [ ] **Step 3: Implement provider protocol and two implementations**

Define `LLMProvider` with `is_in_scope(question: str) -> bool` and `answer(question, chunks) -> str`. The deterministic provider rejects blank questions and returns a stable answer built from the highest-ranked chunk. The OpenAI provider validates its key at construction and calls `client.responses.create` with instructions that require using only supplied context and returning “证据不足” when context cannot answer.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/llm/test_provider.py -v`

Expected: `2 passed`, with no external API request.

- [ ] **Step 5: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/llm projects/enterprise-knowledge-agent/tests/llm
git commit -m "feat: add swappable llm providers"
```

## Task 8: LangGraph RAG and escalation workflow

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/agent/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/agent/state.py`
- Create: `projects/enterprise-knowledge-agent/app/agent/workflow.py`
- Create: `projects/enterprise-knowledge-agent/tests/agent/test_workflow.py`

- [ ] **Step 1: Write failing high-evidence and low-evidence tests**

Use in-memory vector storage, deterministic LLM, SQLite repositories, and a compiled graph. Index one policy chunk. Assert that a matching question returns `status == "answered"`, a citation, and no ticket. Ask an unrelated question against an empty tenant and assert `status == "escalated"` with a persisted open ticket.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/agent/test_workflow.py -v`

Expected: FAIL because Agent workflow modules do not exist.

- [ ] **Step 3: Define typed state**

```python
from typing import TypedDict

from app.retrieval.models import RetrievedChunk


class AgentState(TypedDict, total=False):
    tenant_id: str
    question: str
    in_scope: bool
    chunks: list[RetrievedChunk]
    evidence_ok: bool
    top_score: float
    answer: str
    status: str
    ticket_id: int | None
    started_at: float
```

- [ ] **Step 4: Implement graph nodes and conditional routes**

Build a `KnowledgeAgent` class receiving `VectorStore`, `LLMProvider`, ticket repository, run repository, threshold, and limit. Compile this graph:

```text
START → classify_question
classify_question → retrieve_documents | create_ticket
retrieve_documents → grade_evidence
grade_evidence → generate_answer | create_ticket
generate_answer → record_run
create_ticket → record_run
record_run → END
```

`grade_evidence` sets `top_score` to zero for no results. `generate_answer` maps retrieved chunks directly to citations in the final result. `record_run` persists route, score, ticket ID, and elapsed milliseconds.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/agent/test_workflow.py -v`

Expected: both branches pass and create one Agent-run row each.

- [ ] **Step 6: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/agent projects/enterprise-knowledge-agent/tests/agent
git commit -m "feat: add rag agent workflow"
```

## Task 9: FastAPI endpoints and dependency construction

**Files:**
- Create: `projects/enterprise-knowledge-agent/app/api/__init__.py`
- Create: `projects/enterprise-knowledge-agent/app/api/schemas.py`
- Create: `projects/enterprise-knowledge-agent/app/api/dependencies.py`
- Create: `projects/enterprise-knowledge-agent/app/api/routes.py`
- Create: `projects/enterprise-knowledge-agent/app/main.py`
- Create: `projects/enterprise-knowledge-agent/tests/api/test_api.py`

- [ ] **Step 1: Write failing API behavior tests**

Create the app with dependency overrides for deterministic LLM, in-memory vector storage, and SQLite. Test:

- `GET /health` returns 200 and dependency statuses.
- unsupported upload returns 422.
- supported upload returns document ID and chunk count.
- a supported question returns the discriminated `answered` response with citations.
- an unanswered question returns the `escalated` response with ticket ID.
- `GET /api/v1/tickets/{id}` returns the created ticket.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/api/test_api.py -v`

Expected: FAIL because API modules do not exist.

- [ ] **Step 3: Implement Pydantic schemas**

Define constrained `tenant_id`, nonblank question, ingestion response, citation, question response, ticket response, dependency health, and overall health models. Use `Literal["answered", "escalated"]` for result status.

- [ ] **Step 4: Implement routes and app factory**

Use `/api/v1` router prefix. Limit uploaded bytes before parsing. Convert domain errors to 422 responses and configuration/dependency failures to sanitized 503 responses. Assign an `X-Request-ID` when missing and return it in the response header. Keep routes thin by calling document and Agent services.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/api/test_api.py -v`

Expected: all API tests pass.

- [ ] **Step 6: Run the complete test suite**

Run: `python -m pytest -v`

Expected: all tests pass with no external model calls.

- [ ] **Step 7: Commit**

```bash
git add projects/enterprise-knowledge-agent/app/api projects/enterprise-knowledge-agent/app/main.py projects/enterprise-knowledge-agent/tests/api
git commit -m "feat: expose knowledge agent api"
```

## Task 10: Docker deployment and public sample data

**Files:**
- Create: `projects/enterprise-knowledge-agent/Dockerfile`
- Create: `projects/enterprise-knowledge-agent/compose.yaml`
- Create: `projects/enterprise-knowledge-agent/sample_data/employee_handbook.md`
- Modify: `projects/enterprise-knowledge-agent/.env.example`
- Create: `projects/enterprise-knowledge-agent/tests/test_deployment_files.py`

- [ ] **Step 1: Write failing deployment contract test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compose_declares_required_services_and_healthchecks():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "api:" in compose
    assert "qdrant:" in compose
    assert "postgres:" in compose
    assert compose.count("healthcheck:") >= 3


def test_example_environment_contains_no_real_secret():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in text
    assert "sk-" not in text
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_deployment_files.py -v`

Expected: FAIL because deployment files do not exist.

- [ ] **Step 3: Implement container files**

Use `python:3.11-slim`, install the project from `pyproject.toml`, run as a non-root user, expose port 8000, and start `uvicorn app.main:app --host 0.0.0.0 --port 8000`. Compose uses named volumes, PostgreSQL readiness with `pg_isready`, Qdrant readiness through its HTTP endpoint, API readiness through `/health`, and dependency ordering based on service health.

- [ ] **Step 4: Add safe example data and environment**

The sample employee handbook contains fictional annual-leave, reimbursement, remote-work, and information-security policies. `.env.example` uses deterministic LLM mode by default and documents the PostgreSQL/Qdrant container URLs plus blank `OPENAI_API_KEY`.

- [ ] **Step 5: Verify GREEN and validate Compose**

Run:

```bash
python -m pytest tests/test_deployment_files.py -v
docker compose config
```

Expected: tests pass and Compose exits 0 with normalized configuration.

- [ ] **Step 6: Start and smoke-test the stack**

Run:

```bash
docker compose up --build -d
curl http://localhost:8000/health
docker compose ps
```

Expected: all three services are healthy and `/health` returns HTTP 200.

- [ ] **Step 7: Commit**

```bash
git add projects/enterprise-knowledge-agent/Dockerfile projects/enterprise-knowledge-agent/compose.yaml projects/enterprise-knowledge-agent/.env.example projects/enterprise-knowledge-agent/sample_data projects/enterprise-knowledge-agent/tests/test_deployment_files.py
git commit -m "build: add local production-like stack"
```

## Task 11: Teaching and interview documentation

**Files:**
- Create: `projects/enterprise-knowledge-agent/README.md`
- Create: `projects/enterprise-knowledge-agent/docs/learning-guide.md`
- Create: `projects/enterprise-knowledge-agent/docs/interview-guide.md`
- Create: `projects/enterprise-knowledge-agent/docs/architecture.md`
- Create: `projects/enterprise-knowledge-agent/tests/test_documentation.py`

- [ ] **Step 1: Write failing documentation contract test**

Test that README contains install, test, local run, Docker run, upload, question, and troubleshooting commands; the learning guide contains C# comparisons and exercises; the interview guide contains a 60-second introduction, architecture explanation, tradeoffs, limits, and follow-up questions; the architecture document contains Mermaid diagrams for ingestion and Agent workflow.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_documentation.py -v`

Expected: FAIL because documentation files do not exist.

- [ ] **Step 3: Write README and architecture documentation**

Document commands that have been executed successfully in this project. Include curl examples for document upload and both question branches. Clearly label deterministic mode as a development/test substitute and OpenAI mode as the real LLM path.

- [ ] **Step 4: Write the C#-to-Python learning guide**

Cover project entry, virtual environments, modules, functions, dataclasses, protocols, type hints, dependency injection, async request handling, context-managed database sessions, pytest, and Docker. For every concept, point to a real project file and include one runnable exercise with an expected result.

- [ ] **Step 5: Write the interview guide**

Include an honest project description, 60-second pitch, RAG pipeline, LangGraph routing explanation, tenant isolation, evidence threshold tradeoff, provider abstraction, retry/idempotency discussion, deployment explanation, limitations, and ten likely follow-up questions with concise answers.

- [ ] **Step 6: Verify GREEN**

Run: `python -m pytest tests/test_documentation.py -v`

Expected: documentation contract passes.

- [ ] **Step 7: Commit**

```bash
git add projects/enterprise-knowledge-agent/README.md projects/enterprise-knowledge-agent/docs projects/enterprise-knowledge-agent/tests/test_documentation.py
git commit -m "docs: teach and present knowledge agent project"
```

## Task 12: Final verification and handoff

**Files:**
- Modify only files required by verification failures.

- [ ] **Step 1: Run format and static checks**

Run:

```bash
python -m ruff check .
python -m ruff format --check .
```

Expected: both commands exit 0 with no violations.

- [ ] **Step 2: Run all tests from a clean process**

Run: `python -m pytest -q`

Expected: all tests pass with zero failures and no OpenAI request.

- [ ] **Step 3: Rebuild and test containers**

Run:

```bash
docker compose down
docker compose up --build -d
docker compose ps
```

Expected: `api`, `qdrant`, and `postgres` report healthy.

- [ ] **Step 4: Execute the full demonstration path**

Upload `sample_data/employee_handbook.md`; ask a covered annual-leave question and verify citations; ask an uncovered question and verify a ticket ID; fetch that ticket and verify status `open`.

- [ ] **Step 5: Inspect logs for leaked secrets and uncaught exceptions**

Run: `docker compose logs --no-color`

Expected: no API key, traceback, unhandled exception, or raw database password appears.

- [ ] **Step 6: Stop the stack and record final evidence**

Run: `docker compose down`

Expected: containers stop cleanly while named volumes remain recoverable.

- [ ] **Step 7: Commit any verification-only fixes**

```bash
git add projects/enterprise-knowledge-agent
git commit -m "test: verify enterprise knowledge agent"
```
