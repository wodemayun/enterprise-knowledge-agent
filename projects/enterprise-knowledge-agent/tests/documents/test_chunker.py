import pytest

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

    with pytest.raises(ValueError, match="document contains no text"):
        chunker.split("   \n")


def test_chunker_rejects_overlap_not_smaller_than_chunk():
    with pytest.raises(ValueError, match="overlap smaller"):
        TextChunker(chunk_size=10, overlap=10)
