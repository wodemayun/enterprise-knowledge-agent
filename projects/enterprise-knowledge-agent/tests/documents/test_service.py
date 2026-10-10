from app.documents.chunker import TextChunker
from app.documents.parser import DocumentParser
from app.documents.service import DocumentService
from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.vector_store import InMemoryVectorStore


def test_ingestion_indexes_all_chunks_with_tenant_and_source():
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(64))
    service = DocumentService(DocumentParser(), TextChunker(10, 2), store)

    result = service.ingest("tenant-a", "policy.md", b"abcdefghijklmnopqrstuvwxyz")

    assert result.source_name == "policy.md"
    assert result.chunk_count == 4
    found = store.search("tenant-a", "abcdefghij", limit=10)
    assert found
    assert all(item.source_name == "policy.md" for item in found)


def test_ingestion_is_not_visible_to_another_tenant():
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(64))
    service = DocumentService(DocumentParser(), TextChunker(20, 2), store)
    service.ingest("tenant-a", "policy.md", "员工年假为五天".encode())

    assert store.search("tenant-b", "员工年假", limit=10) == []
