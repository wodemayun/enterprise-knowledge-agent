from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    tenant_id: str
    document_id: str
    source_name: str
    chunk_index: int
    text: str
    score: float

