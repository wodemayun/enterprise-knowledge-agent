from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.agent.workflow import KnowledgeAgent
from app.db import Base
from app.llm.provider import DeterministicProvider
from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.vector_store import InMemoryVectorStore, VectorDocument
from app.runs.models import AgentRun
from app.runs.repository import SqlAlchemyRunRepository
from app.tickets.models import Ticket
from app.tickets.repository import SqlAlchemyTicketRepository


def build_agent(session: Session, store: InMemoryVectorStore) -> KnowledgeAgent:
    return KnowledgeAgent(
        vector_store=store,
        llm=DeterministicProvider(),
        tickets=SqlAlchemyTicketRepository(session),
        runs=SqlAlchemyRunRepository(session),
        evidence_threshold=0.3,
        retrieval_limit=4,
    )


def test_high_evidence_question_returns_answer_with_citation():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(64))
    store.add([VectorDocument("doc-1", "tenant-a", "handbook.md", 0, "员工年假为五天")])

    with Session(engine) as session:
        result = build_agent(session, store).ask("tenant-a", "员工年假多少天")
        session.commit()

        assert result["status"] == "answered"
        assert result["ticket_id"] is None
        assert result["chunks"][0].source_name == "handbook.md"
        assert "五天" in result["answer"]
        assert session.scalar(select(func.count()).select_from(AgentRun)) == 1


def test_missing_evidence_creates_open_ticket():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    store = InMemoryVectorStore(DeterministicEmbeddingProvider(64))

    with Session(engine) as session:
        result = build_agent(session, store).ask("tenant-a", "公司的班车几点发车")
        session.commit()

        assert result["status"] == "escalated"
        assert isinstance(result["ticket_id"], int)
        assert result["chunks"] == []
        assert session.scalar(select(func.count()).select_from(Ticket)) == 1
        assert session.scalar(select(func.count()).select_from(AgentRun)) == 1
