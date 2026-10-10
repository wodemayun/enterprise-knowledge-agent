from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    source_name: str
    chunk_count: int

