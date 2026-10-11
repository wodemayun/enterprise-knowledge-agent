from dataclasses import dataclass

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db import Base, create_session_factory
from app.documents.chunker import TextChunker
from app.documents.parser import DocumentParser
from app.documents.service import DocumentService
from app.llm.provider import DeterministicProvider, LLMProvider, OpenAIProvider
from app.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    FastEmbedEmbeddingProvider,
)
from app.retrieval.vector_store import QdrantVectorStore, VectorStore


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    vector_store: VectorStore
    document_service: DocumentService
    llm: LLMProvider
    session_factory: sessionmaker[Session]


def build_container(settings: Settings | None = None) -> AppContainer:
    settings = settings or get_settings()
    embeddings = (
        FastEmbedEmbeddingProvider(settings.embedding_model)
        if settings.embedding_provider == "fastembed"
        else DeterministicEmbeddingProvider()
    )
    client = (
        QdrantClient(":memory:")
        if settings.qdrant_url == ":memory:"
        else QdrantClient(url=settings.qdrant_url)
    )
    vector_store = QdrantVectorStore(client, settings.qdrant_collection, embeddings)
    session_factory = create_session_factory(settings.database_url)
    Base.metadata.create_all(bind=session_factory.kw["bind"])
    llm = (
        OpenAIProvider(settings.openai_api_key, settings.openai_model)
        if settings.llm_provider == "openai"
        else DeterministicProvider()
    )
    return AppContainer(
        settings=settings,
        vector_store=vector_store,
        document_service=DocumentService(
            DocumentParser(),
            TextChunker(settings.chunk_size, settings.chunk_overlap),
            vector_store,
        ),
        llm=llm,
        session_factory=session_factory,
    )
