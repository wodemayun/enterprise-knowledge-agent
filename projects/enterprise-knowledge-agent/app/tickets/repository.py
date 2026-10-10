from sqlalchemy.orm import Session

from app.tickets.models import Ticket


class SqlAlchemyTicketRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, tenant_id: str, question: str, reason: str) -> Ticket:
        ticket = Ticket(tenant_id=tenant_id, question=question, reason=reason, status="open")
        self._session.add(ticket)
        self._session.flush()
        return ticket

    def get(self, ticket_id: int) -> Ticket | None:
        return self._session.get(Ticket, ticket_id)

