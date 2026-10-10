from typing import TypedDict

from app.retrieval.models import RetrievedChunk


class AgentState(TypedDict, total=False):
    tenant_id: str
    question: str
    in_scope: bool
    chunks: list[RetrievedChunk]
    evidence_ok: bool
    top_score: float
    answer: str
    status: str
    ticket_id: int | None
    started_at: float

