"""Tests for LangGraph Agent State schema."""

from datetime import date

import pytest

from app.agents.state import (
    AgentCounters,
    AgentFilters,
    AgentMetrics,
    AgentState,
    DraftAnswer,
    FinalResponse,
    GroundingCheckResult,
    GroundingVerdict,
    Intent,
    ParsedQuery,
    RetrievalResult,
    create_initial_state,
)
from app.models.schemas import Citation


class TestIntent:
    """Tests for Intent enum."""

    def test_intent_values(self):
        """Test that all expected intent values exist."""
        assert Intent.POLICY_QUESTION.value == "policy_question"
        assert Intent.PROCEDURE_QUESTION.value == "procedure_question"
        assert Intent.GENERAL_QUESTION.value == "general_question"
        assert Intent.OUT_OF_SCOPE.value == "out_of_scope"


class TestGroundingVerdict:
    """Tests for GroundingVerdict enum."""

    def test_verdict_values(self):
        """Test that all expected verdict values exist."""
        assert GroundingVerdict.PASS.value == "pass"
        assert GroundingVerdict.INSUFFICIENT.value == "insufficient"
        assert GroundingVerdict.CONFLICT.value == "conflict"
        assert GroundingVerdict.OFF_TOPIC.value == "off_topic"


class TestParsedQuery:
    """Tests for ParsedQuery dataclass."""

    def test_create_parsed_query(self):
        """Test creating a ParsedQuery."""
        parsed = ParsedQuery(
            normalized_query="환불 정책이 뭐예요?",
            intent=Intent.POLICY_QUESTION,
            confidence=0.95,
            extracted_entities={"category": "refund"},
        )
        assert parsed.normalized_query == "환불 정책이 뭐예요?"
        assert parsed.intent == Intent.POLICY_QUESTION
        assert parsed.confidence == 0.95
        assert parsed.extracted_entities == {"category": "refund"}

    def test_parsed_query_default_entities(self):
        """Test ParsedQuery with default empty entities."""
        parsed = ParsedQuery(
            normalized_query="test",
            intent=Intent.GENERAL_QUESTION,
            confidence=0.5,
        )
        assert parsed.extracted_entities == {}


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_retrieval_result(self):
        """Test creating a RetrievalResult."""
        citation = Citation(
            doc_id="doc_001",
            title="Test Document",
            chunk_id="chunk_001",
            snippet="Test snippet",
            score=0.9,
        )
        result = RetrievalResult(
            candidates=[citation],
            query_embedding=[0.1, 0.2, 0.3],
            total_retrieved=10,
            filter_applied=True,
        )
        assert len(result.candidates) == 1
        assert result.total_retrieved == 10
        assert result.filter_applied is True

    def test_retrieval_result_defaults(self):
        """Test RetrievalResult default values."""
        result = RetrievalResult(candidates=[])
        assert result.query_embedding is None
        assert result.total_retrieved == 0
        assert result.filter_applied is False


class TestDraftAnswer:
    """Tests for DraftAnswer dataclass."""

    def test_create_draft_answer(self):
        """Test creating a DraftAnswer."""
        draft = DraftAnswer(
            content="환불은 7일 이내 가능합니다.",
            citations_used=["chunk_001", "chunk_002"],
            model="gpt-4o-mini",
        )
        assert draft.content == "환불은 7일 이내 가능합니다."
        assert len(draft.citations_used) == 2
        assert draft.model == "gpt-4o-mini"

    def test_draft_answer_default_model(self):
        """Test DraftAnswer default model."""
        draft = DraftAnswer(content="test", citations_used=[])
        assert draft.model == "gpt-4o-mini"


class TestGroundingCheckResult:
    """Tests for GroundingCheckResult dataclass."""

    def test_create_grounding_result_pass(self):
        """Test creating a passing grounding result."""
        result = GroundingCheckResult(
            verdict=GroundingVerdict.PASS,
            evidence_coverage=0.95,
            issues=[],
        )
        assert result.verdict == GroundingVerdict.PASS
        assert result.evidence_coverage == 0.95
        assert len(result.issues) == 0

    def test_create_grounding_result_insufficient(self):
        """Test creating an insufficient grounding result."""
        result = GroundingCheckResult(
            verdict=GroundingVerdict.INSUFFICIENT,
            evidence_coverage=0.4,
            issues=["근거 부족"],
            recommended_action="retry_retrieval",
        )
        assert result.verdict == GroundingVerdict.INSUFFICIENT
        assert result.recommended_action == "retry_retrieval"


class TestFinalResponse:
    """Tests for FinalResponse dataclass."""

    def test_create_final_response(self):
        """Test creating a FinalResponse."""
        citation = Citation(
            doc_id="doc_001",
            title="Test",
            chunk_id="chunk_001",
            snippet="snippet",
            score=0.9,
        )
        final = FinalResponse(
            answer="환불은 가능합니다.",
            citations=[citation],
            follow_up_question=None,
            withheld=False,
        )
        assert final.answer == "환불은 가능합니다."
        assert len(final.citations) == 1
        assert final.withheld is False

    def test_final_response_withheld(self):
        """Test creating a withheld FinalResponse."""
        final = FinalResponse(
            answer="답변을 드리기 어렵습니다.",
            citations=[],
            follow_up_question="질문을 구체적으로 해주세요.",
            withheld=True,
            withheld_reason="근거 부족",
        )
        assert final.withheld is True
        assert final.withheld_reason == "근거 부족"
        assert final.follow_up_question is not None


class TestAgentCounters:
    """Tests for AgentCounters dataclass."""

    def test_default_counters(self):
        """Test default counter values."""
        counters = AgentCounters()
        assert counters.retrieval_attempts == 0
        assert counters.generation_attempts == 0

    def test_increment_counters(self):
        """Test incrementing counters."""
        counters = AgentCounters()
        counters.retrieval_attempts += 1
        assert counters.retrieval_attempts == 1


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = AgentMetrics()
        assert metrics.parse_latency_ms == 0.0
        assert metrics.retrieve_latency_ms == 0.0
        assert metrics.generate_latency_ms == 0.0
        assert metrics.grounding_latency_ms == 0.0
        assert metrics.total_latency_ms == 0.0


class TestAgentFilters:
    """Tests for AgentFilters dataclass."""

    def test_default_filters(self):
        """Test default filter values."""
        filters = AgentFilters()
        assert filters.category is None
        assert filters.store_type is None
        assert filters.effective_date is None
        assert filters.language is None

    def test_filters_with_values(self):
        """Test filters with values."""
        filters = AgentFilters(
            category="refund",
            store_type="cafe",
            effective_date=date(2024, 1, 1),
            language="ko",
        )
        assert filters.category == "refund"
        assert filters.store_type == "cafe"
        assert filters.effective_date == date(2024, 1, 1)


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_basic_state(self):
        """Test creating basic initial state."""
        state = create_initial_state(
            question="환불 정책이 뭐예요?",
            trace_id="test_trace_001",
        )
        assert state["question"] == "환불 정책이 뭐예요?"
        assert state["trace_id"] == "test_trace_001"
        assert state["topk"] == 8  # default
        assert state["counters"].retrieval_attempts == 0
        assert state["errors"] == []

    def test_create_state_with_filters(self):
        """Test creating state with filters."""
        state = create_initial_state(
            question="test",
            trace_id="test_001",
            topk=10,
            store_type="cafe",
            category="refund",
            effective_date="2024-01-15",
            language="ko",
        )
        assert state["topk"] == 10
        assert state["filters"].store_type == "cafe"
        assert state["filters"].category == "refund"
        assert state["filters"].effective_date == date(2024, 1, 15)
        assert state["filters"].language == "ko"

    def test_create_state_with_invalid_date(self):
        """Test creating state with invalid date format."""
        state = create_initial_state(
            question="test",
            trace_id="test_001",
            effective_date="invalid-date",
        )
        # Invalid date should result in None
        assert state["filters"].effective_date is None
