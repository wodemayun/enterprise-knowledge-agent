from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db import Base
from app.runs.models import AgentRun
from app.runs.repository import SqlAlchemyRunRepository
from app.tickets.repository import SqlAlchemyTicketRepository


def test_ticket_and_run_are_persisted():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        tickets = SqlAlchemyTicketRepository(session)
        runs = SqlAlchemyRunRepository(session)
        ticket = tickets.create("tenant-a", "没有答案的问题", "insufficient_evidence")
        run = runs.create(
            "tenant-a",
            "没有答案的问题",
            "escalated",
            0.2,
            ticket.id,
            18,
        )
        session.commit()

        assert tickets.get(ticket.id).status == "open"
        assert run.ticket_id == ticket.id
        assert session.scalar(select(func.count()).select_from(AgentRun)) == 1


def test_missing_ticket_returns_none():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        assert SqlAlchemyTicketRepository(session).get(999) is None
