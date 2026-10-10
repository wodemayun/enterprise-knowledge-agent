from typing import Literal

from pydantic import BaseModel, Field


TenantId = str


class IngestionResponse(BaseModel):
    document_id: str
    source_name: str
    chunk_count: int


class QuestionRequest(BaseModel):
    tenant_id: TenantId = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1, max_length=4000)


class CitationResponse(BaseModel):
    source_name: str
    chunk_index: int
    text: str
    score: float


class QuestionResponse(BaseModel):
    status: Literal["answered", "escalated"]
    answer: str
    citations: list[CitationResponse]
    ticket_id: int | None


class TicketResponse(BaseModel):
    id: int
    tenant_id: str
    question: str
    reason: str
    status: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "unavailable"]
    vector_store: Literal["ok", "unavailable"]

