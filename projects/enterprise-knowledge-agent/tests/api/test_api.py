from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import AppContainer
from app.core.config import Settings
from app.db import Base
from app.documents.chunker import TextChunker
from app.documents.parser import DocumentParser
from app.documents.service import DocumentService
from app.llm.provider import DeterministicProvider
from app.main import create_app
from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.vector_store import InMemoryVectorStore


def make_client() -> TestClient:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    settings = Settings(_env_file=None, evidence_threshold=0.3, chunk_size=100, chunk_overlap=10)
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(64))
    container = AppContainer(
        settings=settings,
        vector_store=store,
        document_service=DocumentService(
            DocumentParser(),
            TextChunker(settings.chunk_size, settings.chunk_overlap),
            store,
        ),
        llm=DeterministicProvider(),
        session_factory=sessions,
    )
    return TestClient(create_app(container))


def test_health_reports_dependencies():
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "vector_store": "ok"}
    assert response.headers["x-request-id"]


def test_upload_question_and_ticket_flow():
    client = make_client()
    upload = client.post(
        "/api/v1/documents",
        data={"tenant_id": "tenant-a"},
        files={"file": ("handbook.md", "员工年假为五天".encode(), "text/markdown")},
    )
    assert upload.status_code == 201
    assert upload.json()["chunk_count"] == 1

    answered = client.post(
        "/api/v1/questions",
        json={"tenant_id": "tenant-a", "question": "员工年假多少天"},
    )
    assert answered.status_code == 200
    assert answered.json()["status"] == "answered"
    assert answered.json()["citations"][0]["source_name"] == "handbook.md"

    escalated = client.post(
        "/api/v1/questions",
        json={"tenant_id": "tenant-b", "question": "班车几点发车"},
    )
    assert escalated.status_code == 200
    body = escalated.json()
    assert body["status"] == "escalated"
    assert body["ticket_id"]

    ticket = client.get(f"/api/v1/tickets/{body['ticket_id']}")
    assert ticket.status_code == 200
    assert ticket.json()["status"] == "open"


def test_unsupported_upload_returns_422():
    response = make_client().post(
        "/api/v1/documents",
        data={"tenant_id": "tenant-a"},
        files={"file": ("handbook.docx", b"content", "application/octet-stream")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "unsupported document type"
