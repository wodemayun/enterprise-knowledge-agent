import pytest

from app.core.errors import EmptyDocumentError, UnsupportedDocumentError
from app.documents.parser import DocumentParser


def test_parser_reads_utf8_text():
    parser = DocumentParser()

    assert parser.parse("handbook.md", "年假制度".encode()) == "年假制度"


def test_parser_rejects_unknown_extension():
    parser = DocumentParser()

    with pytest.raises(UnsupportedDocumentError, match="unsupported document type"):
        parser.parse("handbook.docx", b"content")


def test_parser_rejects_empty_text():
    parser = DocumentParser()

    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        parser.parse("empty.txt", b"  \n")
