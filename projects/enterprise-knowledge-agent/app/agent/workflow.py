import time

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.llm.provider import LLMProvider
from app.retrieval.vector_store import VectorStore
from app.runs.repository import SqlAlchemyRunRepository
from app.tickets.repository import SqlAlchemyTicketRepository


class KnowledgeAgent:
    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMProvider,
        tickets: SqlAlchemyTicketRepository,
        runs: SqlAlchemyRunRepository,
        evidence_threshold: float,
        retrieval_limit: int,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._tickets = tickets
        self._runs = runs
        self._evidence_threshold = evidence_threshold
        self._retrieval_limit = retrieval_limit
        self._graph = self._build_graph()

    def ask(self, tenant_id: str, question: str) -> AgentState:
        return self._graph.invoke(
            AgentState(
                tenant_id=tenant_id,
                question=question,
                chunks=[],
                top_score=0.0,
                ticket_id=None,
                started_at=time.perf_counter(),
            )
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("classify_question", self._classify_question)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("grade_evidence", self._grade_evidence)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("create_ticket", self._create_ticket)
        graph.add_node("record_run", self._record_run)
        graph.add_edge(START, "classify_question")
        graph.add_conditional_edges(
            "classify_question",
            self._route_after_classification,
            {"retrieve": "retrieve_documents", "ticket": "create_ticket"},
        )
        graph.add_edge("retrieve_documents", "grade_evidence")
        graph.add_conditional_edges(
            "grade_evidence",
            self._route_after_evidence,
            {"answer": "generate_answer", "ticket": "create_ticket"},
        )
        graph.add_edge("generate_answer", "record_run")
        graph.add_edge("create_ticket", "record_run")
        graph.add_edge("record_run", END)
        return graph.compile()

    def _classify_question(self, state: AgentState) -> dict[str, object]:
        return {"in_scope": self._llm.is_in_scope(state["question"])}

    @staticmethod
    def _route_after_classification(state: AgentState) -> str:
        return "retrieve" if state["in_scope"] else "ticket"

    def _retrieve_documents(self, state: AgentState) -> dict[str, object]:
        chunks = self._vector_store.search(
            state["tenant_id"],
            state["question"],
            self._retrieval_limit,
        )
        return {"chunks": chunks}

    def _grade_evidence(self, state: AgentState) -> dict[str, object]:
        top_score = state["chunks"][0].score if state["chunks"] else 0.0
        return {
            "top_score": top_score,
            "evidence_ok": bool(state["chunks"]) and top_score >= self._evidence_threshold,
        }

    @staticmethod
    def _route_after_evidence(state: AgentState) -> str:
        return "answer" if state["evidence_ok"] else "ticket"

    def _generate_answer(self, state: AgentState) -> dict[str, object]:
        return {
            "answer": self._llm.answer(state["question"], state["chunks"]),
            "status": "answered",
            "ticket_id": None,
        }

    def _create_ticket(self, state: AgentState) -> dict[str, object]:
        reason = "out_of_scope" if not state.get("in_scope", False) else "insufficient_evidence"
        ticket = self._tickets.create(state["tenant_id"], state["question"], reason)
        return {
            "answer": "知识库中没有足够证据，已创建人工工单。",
            "status": "escalated",
            "ticket_id": ticket.id,
        }

    def _record_run(self, state: AgentState) -> dict[str, object]:
        duration_ms = max(0, round((time.perf_counter() - state["started_at"]) * 1000))
        self._runs.create(
            tenant_id=state["tenant_id"],
            question=state["question"],
            route=state["status"],
            top_score=state.get("top_score", 0.0),
            ticket_id=state.get("ticket_id"),
            duration_ms=duration_ms,
        )
        return {}
