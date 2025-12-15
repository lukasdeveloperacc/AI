"""Tests for LangGraph Agent Graph workflow."""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.graph import (
    MAX_RETRIEVAL_ATTEMPTS,
    build_agent_graph,
    get_agent_graph,
    should_retry_retrieval,
)
from app.agents.state import (
    AgentCounters,
    AgentFilters,
    AgentMetrics,
    AgentState,
    GroundingCheckResult,
    GroundingVerdict,
    create_initial_state,
)


class TestShouldRetryRetrieval:
    """Tests for should_retry_retrieval routing function."""

    def test_retry_when_insufficient_and_attempts_available(self):
        """Test retry when insufficient and attempts < max."""
        state: AgentState = {
            "trace_id": "test",
            "question": "test",
            "topk": 8,
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.INSUFFICIENT,
                evidence_coverage=0.4,
                recommended_action="retry_retrieval",
            ),
            "counters": AgentCounters(retrieval_attempts=1),
            "errors": [],
            "filters": AgentFilters(),
            "metrics": AgentMetrics(),
        }
        result = should_retry_retrieval(state)
        assert result == "retrieve"

    def test_no_retry_when_max_attempts_reached(self):
        """Test no retry when max attempts reached."""
        state: AgentState = {
            "trace_id": "test",
            "question": "test",
            "topk": 8,
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.INSUFFICIENT,
                evidence_coverage=0.4,
                recommended_action="retry_retrieval",
            ),
            "counters": AgentCounters(retrieval_attempts=MAX_RETRIEVAL_ATTEMPTS),
            "errors": [],
            "filters": AgentFilters(),
            "metrics": AgentMetrics(),
        }
        result = should_retry_retrieval(state)
        assert result == "finalize"

    def test_no_retry_when_pass(self):
        """Test no retry when grounding passes."""
        state: AgentState = {
            "trace_id": "test",
            "question": "test",
            "topk": 8,
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.PASS,
                evidence_coverage=0.95,
            ),
            "counters": AgentCounters(retrieval_attempts=1),
            "errors": [],
            "filters": AgentFilters(),
            "metrics": AgentMetrics(),
        }
        result = should_retry_retrieval(state)
        assert result == "finalize"

    def test_no_retry_when_conflict(self):
        """Test no retry when there's a conflict."""
        state: AgentState = {
            "trace_id": "test",
            "question": "test",
            "topk": 8,
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.CONFLICT,
                evidence_coverage=0.8,
            ),
            "counters": AgentCounters(retrieval_attempts=1),
            "errors": [],
            "filters": AgentFilters(),
            "metrics": AgentMetrics(),
        }
        result = should_retry_retrieval(state)
        assert result == "finalize"

    def test_no_retry_when_off_topic(self):
        """Test no retry when query is off-topic."""
        state: AgentState = {
            "trace_id": "test",
            "question": "test",
            "topk": 8,
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.OFF_TOPIC,
                evidence_coverage=0.0,
            ),
            "counters": AgentCounters(retrieval_attempts=1),
            "errors": [],
            "filters": AgentFilters(),
            "metrics": AgentMetrics(),
        }
        result = should_retry_retrieval(state)
        assert result == "finalize"

    def test_finalize_when_no_grounding(self):
        """Test finalize when no grounding result."""
        state: AgentState = {
            "trace_id": "test",
            "question": "test",
            "topk": 8,
            "counters": AgentCounters(retrieval_attempts=1),
            "errors": [],
            "filters": AgentFilters(),
            "metrics": AgentMetrics(),
        }
        result = should_retry_retrieval(state)
        assert result == "finalize"

    def test_no_retry_when_action_not_retry(self):
        """Test no retry when recommended action is not retry_retrieval."""
        state: AgentState = {
            "trace_id": "test",
            "question": "test",
            "topk": 8,
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.INSUFFICIENT,
                evidence_coverage=0.4,
                recommended_action="refine_answer",  # Not retry_retrieval
            ),
            "counters": AgentCounters(retrieval_attempts=1),
            "errors": [],
            "filters": AgentFilters(),
            "metrics": AgentMetrics(),
        }
        result = should_retry_retrieval(state)
        assert result == "finalize"


class TestBuildAgentGraph:
    """Tests for build_agent_graph function."""

    def test_graph_compiles(self):
        """Test that graph compiles without errors."""
        graph = build_agent_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        """Test that graph has expected nodes."""
        graph = build_agent_graph()
        # Access the underlying graph nodes
        nodes = graph.nodes
        expected_nodes = [
            "parse_classify",
            "retrieve",
            "generate",
            "grounding_check",
            "finalize",
        ]
        for node_name in expected_nodes:
            assert node_name in nodes or any(
                node_name in str(n) for n in nodes
            ), f"Node {node_name} not found"


class TestGetAgentGraph:
    """Tests for get_agent_graph singleton."""

    def test_singleton_returns_same_instance(self):
        """Test that get_agent_graph returns same instance."""
        graph1 = get_agent_graph()
        graph2 = get_agent_graph()
        assert graph1 is graph2


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_creates_valid_state(self):
        """Test that create_initial_state creates valid state."""
        state = create_initial_state(
            question="환불 정책이 뭐예요?",
            trace_id="test_trace_001",
            topk=10,
            store_type="cafe",
        )

        assert state["question"] == "환불 정책이 뭐예요?"
        assert state["trace_id"] == "test_trace_001"
        assert state["topk"] == 10
        assert state["filters"].store_type == "cafe"
        assert state["counters"].retrieval_attempts == 0
        assert state["errors"] == []
