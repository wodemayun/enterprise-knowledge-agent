from typing import Protocol

from openai import OpenAI

from app.core.errors import ConfigurationError
from app.retrieval.models import RetrievedChunk


class LLMProvider(Protocol):
    def is_in_scope(self, question: str) -> bool: ...

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str: ...


class DeterministicProvider:
    """Predictable provider for tests and offline workflow demonstrations."""

    def is_in_scope(self, question: str) -> bool:
        return bool(question.strip())

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not question.strip() or not chunks:
            return "证据不足，无法根据知识库回答。"
        best = chunks[0]
        return f"根据《{best.source_name}》：{best.text}"


class OpenAIProvider:
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI mode")
        self._client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
        self._model = model

    def is_in_scope(self, question: str) -> bool:
        return bool(question.strip())

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[{chunk.source_name}#{chunk.chunk_index}] {chunk.text}" for chunk in chunks
        )
        response = self._client.responses.create(
            model=self._model,
            instructions=(
                "你是企业知识库助手。只能根据给出的知识库片段回答，不得补充片段中没有的事实。"
                "如果片段无法回答，明确回复‘证据不足’。回答简洁，并保留来源编号。"
            ),
            input=f"问题：{question}\n\n知识库片段：\n{context}",
        )
        return response.output_text
