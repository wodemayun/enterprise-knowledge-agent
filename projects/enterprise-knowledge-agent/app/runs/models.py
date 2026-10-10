from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (Index("ix_runs_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(100), index=True)
    question: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(30))
    top_score: Mapped[float] = mapped_column(Float)
    ticket_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
