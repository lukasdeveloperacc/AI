"""Tests for LangGraph Agent Node functions."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.nodes.retrieve import (
    _apply_filters,
    _is_filter_empty,
    _limit_chunks_per_doc,
    _matches_category,
    _matches_store_type,
)
from app.agents.state import (
    AgentCounters,
    AgentFilters,
    AgentMetrics,
    AgentState,
    DraftAnswer,
    GroundingCheckResult,
    GroundingVerdict,
    Intent,
    ParsedQuery,
    RetrievalResult,
)
from app.models.schemas import Citation


class TestRetrieveNodeHelpers:
    """Tests for retrieve node helper functions."""

    def test_is_filter_empty_true(self):
        """Test is_filter_empty returns True for empty filters."""
        filters = AgentFilters()
        assert _is_filter_empty(filters) is True

    def test_is_filter_empty_false(self):
        """Test is_filter_empty returns False for non-empty filters."""
        filters = AgentFilters(category="refund")
        assert _is_filter_empty(filters) is False

    def test_matches_category_in_doc_id(self):
        """Test matches_category finds category in doc_id."""
        citation = Citation(
            doc_id="refund_policy_001",
            title="General Policy",
            chunk_id="chunk_001",
            snippet="test",
            score=0.9,
        )
        assert _matches_category(citation, "refund") is True
        assert _matches_category(citation, "promo") is False

    def test_matches_category_in_title(self):
        """Test matches_category finds category in title."""
        citation = Citation(
            doc_id="doc_001",
            title="환불 정책 안내",
            chunk_id="chunk_001",
            snippet="test",
            score=0.9,
        )
        assert _matches_category(citation, "환불") is True

    def test_matches_store_type_in_doc_id(self):
        """Test matches_store_type finds store_type in doc_id."""
        citation = Citation(
            doc_id="cafe_manual_001",
            title="운영 매뉴얼",
            chunk_id="chunk_001",
            snippet="test",
            score=0.9,
        )
        assert _matches_store_type(citation, "cafe") is True
        assert _matches_store_type(citation, "convenience") is False

    def test_limit_chunks_per_doc(self):
        """Test limit_chunks_per_doc limits chunks correctly."""
        citations = [
            Citation(doc_id="doc_001", title="T1", chunk_id="c1", snippet="s1", score=0.9),
            Citation(doc_id="doc_001", title="T1", chunk_id="c2", snippet="s2", score=0.85),
            Citation(doc_id="doc_001", title="T1", chunk_id="c3", snippet="s3", score=0.8),
            Citation(doc_id="doc_002", title="T2", chunk_id="c4", snippet="s4", score=0.75),
            Citation(doc_id="doc_002", title="T2", chunk_id="c5", snippet="s5", score=0.7),
        ]
        limited = _limit_chunks_per_doc(citations, max_chunks=2)

        # Should have 2 from doc_001 and 2 from doc_002
        assert len(limited) == 4

        # Count per doc
        doc_counts = {}
        for c in limited:
            doc_counts[c.doc_id] = doc_counts.get(c.doc_id, 0) + 1

        assert doc_counts["doc_001"] == 2
        assert doc_counts["doc_002"] == 2

    def test_apply_filters_empty(self):
        """Test apply_filters with empty filters."""
        citations = [
            Citation(doc_id="doc_001", title="T1", chunk_id="c1", snippet="s1", score=0.9),
        ]
        filters = AgentFilters()
        result = _apply_filters(citations, filters)
        assert len(result) == 1  # No filtering

    def test_apply_filters_category(self):
        """Test apply_filters with category filter."""
        citations = [
            Citation(doc_id="refund_001", title="T1", chunk_id="c1", snippet="s1", score=0.9),
            Citation(doc_id="promo_001", title="T2", chunk_id="c2", snippet="s2", score=0.85),
        ]
        filters = AgentFilters(category="refund")
        result = _apply_filters(citations, filters)
        assert len(result) == 1
        assert result[0].doc_id == "refund_001"


class TestGroundingCheckHelpers:
    """Tests for grounding check related functionality."""

    def test_grounding_verdict_pass(self):
        """Test creating pass verdict."""
        result = GroundingCheckResult(
            verdict=GroundingVerdict.PASS,
            evidence_coverage=0.95,
        )
        assert result.verdict == GroundingVerdict.PASS
        assert result.evidence_coverage == 0.95

    def test_grounding_verdict_insufficient(self):
        """Test creating insufficient verdict."""
        result = GroundingCheckResult(
            verdict=GroundingVerdict.INSUFFICIENT,
            evidence_coverage=0.4,
            issues=["근거 부족", "인용 수 부족"],
            recommended_action="retry_retrieval",
        )
        assert result.verdict == GroundingVerdict.INSUFFICIENT
        assert len(result.issues) == 2
        assert result.recommended_action == "retry_retrieval"

    def test_grounding_verdict_conflict(self):
        """Test creating conflict verdict."""
        result = GroundingCheckResult(
            verdict=GroundingVerdict.CONFLICT,
            evidence_coverage=0.8,
            issues=["문서 간 상충"],
            recommended_action="resolve_conflict",
        )
        assert result.verdict == GroundingVerdict.CONFLICT


class TestFinalizeHelpers:
    """Tests for finalize node functionality."""

    def test_draft_answer_creation(self):
        """Test creating draft answer."""
        draft = DraftAnswer(
            content="환불은 7일 이내 가능합니다.",
            citations_used=["chunk_001", "chunk_002"],
            model="gpt-4o-mini",
        )
        assert "환불" in draft.content
        assert len(draft.citations_used) == 2

    def test_parsed_query_creation(self):
        """Test creating parsed query."""
        parsed = ParsedQuery(
            normalized_query="환불 정책이 뭐예요",
            intent=Intent.POLICY_QUESTION,
            confidence=0.92,
            extracted_entities={"category": "refund"},
        )
        assert parsed.intent == Intent.POLICY_QUESTION
        assert parsed.confidence > 0.9


class TestAgentStateFlow:
    """Tests for agent state flow through nodes."""

    def test_state_accumulation(self):
        """Test that state accumulates through nodes."""
        # Start with initial state
        state: AgentState = {
            "trace_id": "test_001",
            "question": "환불 정책이 뭐예요?",
            "topk": 8,
            "filters": AgentFilters(),
            "counters": AgentCounters(),
            "metrics": AgentMetrics(),
            "errors": [],
        }

        # After parse_classify
        state["parsed"] = ParsedQuery(
            normalized_query="환불 정책이 뭐예요",
            intent=Intent.POLICY_QUESTION,
            confidence=0.9,
        )

        # After retrieve
        state["retrieval"] = RetrievalResult(
            candidates=[
                Citation(
                    doc_id="doc_001",
                    title="환불 정책",
                    chunk_id="chunk_001",
                    snippet="환불은 7일 이내 가능",
                    score=0.92,
                )
            ],
            total_retrieved=10,
        )

        # After generate
        state["draft"] = DraftAnswer(
            content="환불은 7일 이내 가능합니다.",
            citations_used=["chunk_001"],
        )

        # After grounding
        state["grounding"] = GroundingCheckResult(
            verdict=GroundingVerdict.PASS,
            evidence_coverage=0.95,
        )

        # Verify all state is present
        assert state["parsed"] is not None
        assert state["retrieval"] is not None
        assert state["draft"] is not None
        assert state["grounding"] is not None
        assert state["grounding"].verdict == GroundingVerdict.PASS

    def test_counter_tracking(self):
        """Test that counters track attempts correctly."""
        counters = AgentCounters()
        assert counters.retrieval_attempts == 0

        # Simulate first retrieval
        counters.retrieval_attempts += 1
        assert counters.retrieval_attempts == 1

        # Simulate retry
        counters.retrieval_attempts += 1
        assert counters.retrieval_attempts == 2

    def test_metrics_tracking(self):
        """Test that metrics track latency correctly."""
        metrics = AgentMetrics()

        # Simulate node latencies
        metrics.parse_latency_ms = 50.0
        metrics.retrieve_latency_ms = 100.0
        metrics.generate_latency_ms = 200.0
        metrics.grounding_latency_ms = 75.0

        # Calculate total
        metrics.total_latency_ms = (
            metrics.parse_latency_ms
            + metrics.retrieve_latency_ms
            + metrics.generate_latency_ms
            + metrics.grounding_latency_ms
        )

        assert metrics.total_latency_ms == 425.0
