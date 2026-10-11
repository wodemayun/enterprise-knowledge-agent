import math
import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient, models

from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class VectorDocument:
    document_id: str
    tenant_id: str
    source_name: str
    chunk_index: int
    text: str


class VectorStore(Protocol):
    def add(self, documents: list[VectorDocument]) -> None: ...

    def search(self, tenant_id: str, query: str, limit: int) -> list[RetrievedChunk]: ...

    def healthcheck(self) -> bool: ...


class InMemoryVectorStore:
    def __init__(self, embeddings: EmbeddingProvider) -> None:
        self._embeddings = embeddings
        self._items: list[tuple[VectorDocument, list[float]]] = []

    def add(self, documents: list[VectorDocument]) -> None:
        vectors = self._embeddings.embed([document.text for document in documents])
        self._items.extend(zip(documents, vectors, strict=True))

    def search(self, tenant_id: str, query: str, limit: int) -> list[RetrievedChunk]:
        query_vector = self._embeddings.embed([query])[0]
        matches = [
            RetrievedChunk(
                tenant_id=document.tenant_id,
                document_id=document.document_id,
                source_name=document.source_name,
                chunk_index=document.chunk_index,
                text=document.text,
                score=self._cosine(query_vector, vector),
            )
            for document, vector in self._items
            if document.tenant_id == tenant_id
        ]
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def healthcheck(self) -> bool:
        return True

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: EmbeddingProvider,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embeddings = embeddings

    def add(self, documents: list[VectorDocument]) -> None:
        if not documents:
            return
        vectors = self._embeddings.embed([document.text for document in documents])
        self._ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "document_id": document.document_id,
                    "tenant_id": document.tenant_id,
                    "source_name": document.source_name,
                    "chunk_index": document.chunk_index,
                    "text": document.text,
                },
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        self._client.upsert(self._collection_name, points=points, wait=True)

    def search(self, tenant_id: str, query: str, limit: int) -> list[RetrievedChunk]:
        if not self._client.collection_exists(self._collection_name):
            return []
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=self._embeddings.embed([query])[0],
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return [self._to_retrieved(point) for point in response.points]

    def healthcheck(self) -> bool:
        self._client.get_collections()
        return True

    def _ensure_collection(self, dimensions: int) -> None:
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=dimensions,
                    distance=models.Distance.COSINE,
                ),
            )

    @staticmethod
    def _to_retrieved(point: models.ScoredPoint) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            tenant_id=str(payload["tenant_id"]),
            document_id=str(payload["document_id"]),
            source_name=str(payload["source_name"]),
            chunk_index=int(payload["chunk_index"]),
            text=str(payload["text"]),
            score=float(point.score),
        )
