import pytest

from app.core.errors import ConfigurationError
from app.llm.provider import DeterministicProvider, OpenAIProvider
from app.retrieval.models import RetrievedChunk


def test_deterministic_provider_answers_from_context():
    provider = DeterministicProvider()
    chunk = RetrievedChunk("t", "d", "policy.md", 0, "年假为五天", 0.9)

    answer = provider.answer("年假多少天？", [chunk])

    assert "年假为五天" in answer
    assert provider.is_in_scope("年假多少天？") is True


def test_deterministic_provider_rejects_blank_question():
    assert DeterministicProvider().is_in_scope("  ") is False


def test_openai_provider_requires_key():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIProvider(api_key=None, model="gpt-5.5")
