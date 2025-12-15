"""LangGraph State schema for StoreOps Agent.

This module defines the state schema for the RAG agent workflow:
Parse/Classify -> Retrieve -> Generate -> Grounding Check -> Finalize
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional

from typing_extensions import TypedDict

from app.models.schemas import Citation, Verdict


class Intent(str, Enum):
    """Intent classification for user queries."""

    POLICY_QUESTION = "policy_question"  # 규정/정책 관련 질문
    PROCEDURE_QUESTION = "procedure_question"  # 절차/프로세스 질문
    GENERAL_QUESTION = "general_question"  # 일반적인 질문
    OUT_OF_SCOPE = "out_of_scope"  # 범위 외 질문


class GroundingVerdict(str, Enum):
    """Verdict from grounding check."""

    PASS = "pass"  # 충분한 근거로 답변 가능
    INSUFFICIENT = "insufficient"  # 근거 부족
    CONFLICT = "conflict"  # 문서 간 충돌
    OFF_TOPIC = "off_topic"  # 주제와 무관


@dataclass
class ParsedQuery:
    """Result of query parsing and classification.

    Attributes:
        normalized_query: Normalized/cleaned query string.
        intent: Classified intent of the query.
        confidence: Confidence score for classification (0-1).
        extracted_entities: Extracted entities from query.
    """

    normalized_query: str
    intent: Intent
    confidence: float
    extracted_entities: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of document retrieval.

    Attributes:
        candidates: List of retrieved citations.
        query_embedding: Embedding vector used for search.
        total_retrieved: Total number of documents retrieved before filtering.
        filter_applied: Whether metadata filtering was applied.
    """

    candidates: list[Citation]
    query_embedding: Optional[list[float]] = None
    total_retrieved: int = 0
    filter_applied: bool = False


@dataclass
class DraftAnswer:
    """Draft answer before grounding check.

    Attributes:
        content: The draft answer text.
        citations_used: List of citation IDs used.
        model: Model used for generation.
    """

    content: str
    citations_used: list[str]
    model: str = "gpt-4o-mini"


@dataclass
class GroundingCheckResult:
    """Result of grounding check.

    Attributes:
        verdict: Pass/insufficient/conflict/off_topic.
        evidence_coverage: Percentage of answer supported by evidence.
        issues: List of issues found.
        recommended_action: Suggested action based on verdict.
    """

    verdict: GroundingVerdict
    evidence_coverage: float = 0.0
    issues: list[str] = field(default_factory=list)
    recommended_action: Optional[str] = None


@dataclass
class FinalResponse:
    """Final response to return to user.

    Attributes:
        answer: Final answer text.
        citations: Supporting citations.
        follow_up_question: Optional follow-up question if evidence insufficient.
        withheld: Whether answer was withheld.
        withheld_reason: Reason for withholding answer.
    """

    answer: str
    citations: list[Citation]
    follow_up_question: Optional[str] = None
    withheld: bool = False
    withheld_reason: Optional[str] = None


@dataclass
class AgentCounters:
    """Counters for loop control and metrics.

    Attributes:
        retrieval_attempts: Number of retrieval attempts (max 2).
        generation_attempts: Number of generation attempts.
    """

    retrieval_attempts: int = 0
    generation_attempts: int = 0


@dataclass
class AgentMetrics:
    """Performance metrics for the agent run.

    Attributes:
        parse_latency_ms: Time spent in parse/classify node.
        retrieve_latency_ms: Time spent in retrieve node.
        generate_latency_ms: Time spent in generate node.
        grounding_latency_ms: Time spent in grounding check node.
        total_latency_ms: Total processing time.
    """

    parse_latency_ms: float = 0.0
    retrieve_latency_ms: float = 0.0
    generate_latency_ms: float = 0.0
    grounding_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


@dataclass
class AgentFilters:
    """Metadata filters extracted or provided.

    Attributes:
        category: Document category filter.
        store_type: Store type filter.
        effective_date: Date filter for document validity.
        language: Language filter.
    """

    category: Optional[str] = None
    store_type: Optional[str] = None
    effective_date: Optional[date] = None
    language: Optional[str] = None


class AgentState(TypedDict, total=False):
    """State schema for the StoreOps RAG Agent.

    This state flows through the agent graph:
    START -> parse_classify -> retrieve -> generate -> grounding_check -> finalize -> END

    With conditional routing:
    - If grounding_check verdict is 'insufficient' and retrieval_attempts < 2:
      -> back to retrieve (with relaxed filters)
    - Otherwise: -> finalize

    Attributes:
        trace_id: Unique identifier for request tracing.
        question: Original user question.
        filters: Metadata filters (from request or extracted).
        parsed: Result of parse/classify node.
        retrieval: Result of retrieve node.
        draft: Draft answer from generate node.
        grounding: Result of grounding check.
        final: Final response to return.
        counters: Loop control counters.
        metrics: Performance metrics.
        errors: List of errors encountered.
    """

    # Request context
    trace_id: str
    question: str
    topk: int

    # Filters (from request or extracted)
    filters: AgentFilters

    # Node outputs
    parsed: ParsedQuery
    retrieval: RetrievalResult
    draft: DraftAnswer
    grounding: GroundingCheckResult
    final: FinalResponse

    # Control flow
    counters: AgentCounters
    metrics: AgentMetrics
    errors: list[str]


def create_initial_state(
    question: str,
    trace_id: str,
    topk: int = 8,
    store_type: Optional[str] = None,
    category: Optional[str] = None,
    effective_date: Optional[str] = None,
    language: Optional[str] = None,
) -> AgentState:
    """Create initial state for a new agent run.

    Args:
        question: User's question.
        trace_id: Unique trace ID for the request.
        topk: Number of documents to retrieve.
        store_type: Optional store type filter.
        category: Optional category filter.
        effective_date: Optional effective date filter (YYYY-MM-DD).
        language: Optional language filter.

    Returns:
        Initial AgentState ready for graph execution.
    """
    from datetime import datetime

    parsed_date = None
    if effective_date:
        try:
            parsed_date = datetime.strptime(effective_date, "%Y-%m-%d").date()
        except ValueError:
            pass

    return AgentState(
        trace_id=trace_id,
        question=question,
        topk=topk,
        filters=AgentFilters(
            category=category,
            store_type=store_type,
            effective_date=parsed_date,
            language=language,
        ),
        counters=AgentCounters(),
        metrics=AgentMetrics(),
        errors=[],
    )
