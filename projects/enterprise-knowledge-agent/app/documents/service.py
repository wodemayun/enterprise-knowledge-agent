import uuid

from app.documents.chunker import TextChunker
from app.documents.models import IngestionResult
from app.documents.parser import DocumentParser
from app.retrieval.vector_store import VectorDocument, VectorStore


class DocumentService:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: TextChunker,
        vector_store: VectorStore,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._vector_store = vector_store

    def ingest(self, tenant_id: str, filename: str, content: bytes) -> IngestionResult:
        text = self._parser.parse(filename, content)
        chunks = self._chunker.split(text)
        document_id = str(uuid.uuid4())
        self._vector_store.add(
            [
                VectorDocument(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    source_name=filename,
                    chunk_index=chunk.index,
                    text=chunk.text,
                )
                for chunk in chunks
            ]
        )
        return IngestionResult(
            document_id=document_id,
            source_name=filename,
            chunk_count=len(chunks),
        )
