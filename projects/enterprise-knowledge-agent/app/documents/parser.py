from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.core.errors import EmptyDocumentError, UnsupportedDocumentError


class DocumentParser:
    def parse(self, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md"}:
            text = content.decode("utf-8-sig")
        elif suffix == ".pdf":
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            raise UnsupportedDocumentError("unsupported document type")
        if not text.strip():
            raise EmptyDocumentError("document contains no extractable text")
        return text
