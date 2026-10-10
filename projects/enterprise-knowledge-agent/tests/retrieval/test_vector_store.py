import pytest
from qdrant_client import QdrantClient

from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorDocument,
)


@pytest.fixture
def documents():
    return [
        VectorDocument("a-1", "tenant-a", "a.md", 0, "员工年假为五天"),
        VectorDocument("b-1", "tenant-b", "b.md", 0, "员工年假为十五天"),
    ]


def assert_tenant_isolation(store, documents):
    store.add(documents)

    results = store.search("tenant-a", "员工年假多少天", limit=5)

    assert results
    assert {result.tenant_id for result in results} == {"tenant-a"}
    assert results[0].source_name == "a.md"


def test_in_memory_search_never_returns_another_tenants_chunks(documents):
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(dimensions=64))

    assert_tenant_isolation(store, documents)


def test_qdrant_search_never_returns_another_tenants_chunks(documents):
    store = QdrantVectorStore(
        client=QdrantClient(":memory:"),
        collection_name="test_chunks",
        embeddings=DeterministicEmbeddingProvider(dimensions=64),
    )

    assert_tenant_isolation(store, documents)
