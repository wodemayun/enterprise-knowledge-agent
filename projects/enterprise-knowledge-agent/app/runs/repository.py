from sqlalchemy.orm import Session

from app.runs.models import AgentRun


class SqlAlchemyRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        tenant_id: str,
        question: str,
        route: str,
        top_score: float,
        ticket_id: int | None,
        duration_ms: int,
    ) -> AgentRun:
        run = AgentRun(
            tenant_id=tenant_id,
            question=question,
            route=route,
            top_score=top_score,
            ticket_id=ticket_id,
            duration_ms=duration_ms,
        )
        self._session.add(run)
        self._session.flush()
        return run
