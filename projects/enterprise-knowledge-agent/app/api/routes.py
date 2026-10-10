from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import text

from app.agent.workflow import KnowledgeAgent
from app.api.dependencies import AppContainer
from app.api.schemas import (
    CitationResponse,
    HealthResponse,
    IngestionResponse,
    QuestionRequest,
    QuestionResponse,
    TicketResponse,
)
from app.runs.repository import SqlAlchemyRunRepository
from app.tickets.repository import SqlAlchemyTicketRepository

router = APIRouter()


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    container = get_container(request)
    database = "ok"
    vector_store = "ok"
    try:
        with container.session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        database = "unavailable"
    try:
        container.vector_store.healthcheck()
    except Exception:
        vector_store = "unavailable"
    overall = "ok" if database == vector_store == "ok" else "degraded"
    return HealthResponse(status=overall, database=database, vector_store=vector_store)


@router.post(
    "/api/v1/documents",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    request: Request,
    tenant_id: Annotated[str, Form(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")],
    file: Annotated[UploadFile, File()],
) -> IngestionResponse:
    container = get_container(request)
    content = await file.read(container.settings.max_upload_bytes + 1)
    if len(content) > container.settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="uploaded file is too large")
    result = container.document_service.ingest(tenant_id, file.filename or "document", content)
    return IngestionResponse(
        document_id=result.document_id,
        source_name=result.source_name,
        chunk_count=result.chunk_count,
    )


@router.post("/api/v1/questions", response_model=QuestionResponse)
def ask_question(request: Request, payload: QuestionRequest) -> QuestionResponse:
    container = get_container(request)
    with container.session_factory() as session:
        try:
            agent = KnowledgeAgent(
                vector_store=container.vector_store,
                llm=container.llm,
                tickets=SqlAlchemyTicketRepository(session),
                runs=SqlAlchemyRunRepository(session),
                evidence_threshold=container.settings.evidence_threshold,
                retrieval_limit=container.settings.retrieval_limit,
            )
            result = agent.ask(payload.tenant_id, payload.question)
            session.commit()
        except Exception:
            session.rollback()
            raise
    citations = [
        CitationResponse(
            source_name=chunk.source_name,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=chunk.score,
        )
        for chunk in result["chunks"]
        if result["status"] == "answered"
    ]
    return QuestionResponse(
        status=result["status"],
        answer=result["answer"],
        citations=citations,
        ticket_id=result.get("ticket_id"),
    )


@router.get("/api/v1/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(request: Request, ticket_id: int) -> TicketResponse:
    container = get_container(request)
    with container.session_factory() as session:
        ticket = SqlAlchemyTicketRepository(session).get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return TicketResponse(
            id=ticket.id,
            tenant_id=ticket.tenant_id,
            question=ticket.question,
            reason=ticket.reason,
            status=ticket.status,
        )
