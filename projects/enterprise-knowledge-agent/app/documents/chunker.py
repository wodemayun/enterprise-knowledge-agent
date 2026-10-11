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
