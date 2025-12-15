"""Tests for the POST /chat endpoint."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.agents.state import (
    AgentCounters,
    AgentFilters,
    AgentMetrics,
    FinalResponse,
    GroundingCheckResult,
    GroundingVerdict,
)
from app.models.schemas import Citation


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent_state():
    """Create a mock agent state for testing."""
    return {
        "trace_id": "tr_test123",
        "question": "환불 정책이 어떻게 되나요?",
        "topk": 8,
        "filters": AgentFilters(
            category="refund",
            store_type="cafe",
        ),
        "final": FinalResponse(
            answer="환불은 구매 후 7일 이내에 가능합니다.",
            citations=[
                Citation(
                    doc_id="doc_001",
                    title="환불 정책 가이드",
                    chunk_id="chunk_001",
                    snippet="구매 후 7일 이내 환불 가능",
                    score=0.95,
                )
            ],
            follow_up_question="특정 상품에 대해 더 알고 싶으신가요?",
        ),
        "counters": AgentCounters(retrieval_attempts=1, generation_attempts=1),
        "metrics": AgentMetrics(total_latency_ms=500.0),
        "grounding": GroundingCheckResult(
            verdict=GroundingVerdict.PASS,
            evidence_coverage=0.9,
        ),
    }


class TestChatEndpoint:
    """Tests for POST /api/v1/chat endpoint."""

    def test_chat_endpoint_exists(self, client):
        """Test that the chat endpoint exists."""
        response = client.post(
            "/api/v1/chat",
            json={"question": "테스트 질문"},
        )
        # Should not return 404
        assert response.status_code != 404

    @patch("app.api.chat.run_agent")
    def test_chat_success(self, mock_run_agent, client, mock_agent_state):
        """Test successful chat request."""
        mock_run_agent.return_value = mock_agent_state

        response = client.post(
            "/api/v1/chat",
            json={
                "question": "환불 정책이 어떻게 되나요?",
                "store_type": "cafe",
                "category": "refund",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "answer" in data
        assert "citations" in data
        assert "follow_up_question" in data
        assert "meta" in data

        # Check meta fields
        meta = data["meta"]
        assert "trace_id" in meta
        assert "topk" in meta
        assert "retrieval_attempts" in meta
        assert "generation_attempts" in meta
        assert "latency_ms" in meta
        assert "verdict" in meta

    @patch("app.api.chat.run_agent")
    def test_chat_with_citations(self, mock_run_agent, client, mock_agent_state):
        """Test that citations are properly returned."""
        mock_run_agent.return_value = mock_agent_state

        response = client.post(
            "/api/v1/chat",
            json={"question": "환불 정책이 어떻게 되나요?"},
        )

        assert response.status_code == 200
        data = response.json()

        citations = data["citations"]
        assert len(citations) == 1
        assert citations[0]["doc_id"] == "doc_001"
        assert citations[0]["title"] == "환불 정책 가이드"
        assert "score" in citations[0]

    @patch("app.api.chat.run_agent")
    def test_chat_with_filters(self, mock_run_agent, client, mock_agent_state):
        """Test chat with filter parameters."""
        mock_run_agent.return_value = mock_agent_state

        response = client.post(
            "/api/v1/chat",
            json={
                "question": "영업시간이 어떻게 되나요?",
                "store_type": "cafe",
                "category": "operation",
                "language": "ko",
            },
        )

        assert response.status_code == 200

        # Verify run_agent was called with correct parameters
        mock_run_agent.assert_called_once()
        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs["store_type"] == "cafe"
        assert call_kwargs["category"] == "operation"
        assert call_kwargs["language"] == "ko"

    def test_chat_validation_empty_question(self, client):
        """Test validation error for empty question."""
        response = client.post(
            "/api/v1/chat",
            json={"question": ""},
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_chat_validation_missing_question(self, client):
        """Test validation error for missing question field."""
        response = client.post(
            "/api/v1/chat",
            json={},
        )

        assert response.status_code == 422

    def test_chat_response_headers(self, client):
        """Test that response includes expected headers."""
        with patch("app.api.chat.run_agent") as mock_run_agent:
            mock_run_agent.return_value = {
                "trace_id": "tr_test",
                "topk": 8,
                "final": FinalResponse(
                    answer="테스트 답변",
                    citations=[],
                ),
                "counters": AgentCounters(),
                "grounding": None,
                "filters": None,
            }

            response = client.post(
                "/api/v1/chat",
                json={"question": "테스트"},
            )

            # Check for latency header
            assert "x-response-time-ms" in response.headers


class TestChatSchemas:
    """Tests for chat request/response schemas."""

    def test_chat_request_schema(self):
        """Test ChatRequest schema validation."""
        from app.models.schemas import ChatRequest

        # Valid request
        request = ChatRequest(
            question="테스트 질문",
            store_type="cafe",
            category="refund",
        )
        assert request.question == "테스트 질문"
        assert request.store_type == "cafe"

        # Optional fields default to None
        request = ChatRequest(question="질문만")
        assert request.store_type is None
        assert request.category is None

    def test_chat_response_schema(self):
        """Test ChatResponse schema structure."""
        from app.models.schemas import ChatResponse, Meta, Verdict

        response = ChatResponse(
            answer="테스트 답변",
            citations=[],
            follow_up_question="후속 질문",
            meta=Meta(
                trace_id="tr_123",
                topk=8,
                retrieval_attempts=1,
                generation_attempts=1,
                latency_ms=100.0,
                verdict=Verdict.ANSWERED,
            ),
        )

        assert response.answer == "테스트 답변"
        assert response.follow_up_question == "후속 질문"
        assert response.meta.verdict == Verdict.ANSWERED


class TestErrorHandling:
    """Tests for error handling in chat endpoint."""

    def test_error_response_format(self):
        """Test error response schema."""
        from app.models.schemas import ErrorDetail, ErrorResponse

        error = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Invalid request",
                trace_id="tr_123",
            )
        )

        assert error.error.code == "VALIDATION_ERROR"
        assert error.error.trace_id == "tr_123"

    @patch("app.api.chat.run_agent")
    def test_internal_error_includes_trace_id(self, mock_run_agent, client):
        """Test that internal errors include trace_id."""
        mock_run_agent.side_effect = Exception("Test error")

        response = client.post(
            "/api/v1/chat",
            json={"question": "테스트"},
        )

        # Should return 500 with trace_id in header
        assert response.status_code == 500
        assert "x-trace-id" in response.headers
