from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Enterprise Knowledge Agent"
    environment: str = "development"
    database_url: str = "sqlite:///./knowledge_agent.db"
    qdrant_url: str = ":memory:"
    qdrant_collection: str = "knowledge_chunks"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "deterministic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.5"
    evidence_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    retrieval_limit: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
