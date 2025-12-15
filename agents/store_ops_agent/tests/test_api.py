"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to StoreOpsAgent API"}


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestQuestionEndpoint:
    """Tests for question API endpoint."""

    def test_ask_question_success(self, client):
        """Test asking a question returns valid response."""
        response = client.post(
            "/api/v1/ask",
            json={"question": "What are the store hours?"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "meta" in data

        # Check meta fields
        meta = data["meta"]
        assert "trace_id" in meta
        assert meta["trace_id"].startswith("tr_")
        assert "topk" in meta
        assert "retrieval_attempts" in meta
        assert "latency_ms" in meta
        assert "verdict" in meta

    def test_ask_question_with_custom_topk(self, client):
        """Test asking a question with custom topk."""
        response = client.post(
            "/api/v1/ask",
            json={"question": "Test question", "topk": 10},
        )
        assert response.status_code == 200
        assert response.json()["meta"]["topk"] == 10

    def test_ask_question_empty_question(self, client):
        """Test that empty question is rejected."""
        response = client.post(
            "/api/v1/ask",
            json={"question": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_ask_question_invalid_topk(self, client):
        """Test that invalid topk is rejected."""
        response = client.post(
            "/api/v1/ask",
            json={"question": "Test", "topk": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/api/v1/ask",
            json={"question": "Test", "topk": 25},
        )
        assert response.status_code == 422

    def test_ask_question_citations_structure(self, client):
        """Test that citations have correct structure."""
        response = client.post(
            "/api/v1/ask",
            json={"question": "What are the store policies?"},
        )
        assert response.status_code == 200

        data = response.json()
        citations = data["citations"]

        for citation in citations:
            assert "doc_id" in citation
            assert "title" in citation
            assert "chunk_id" in citation
            assert "snippet" in citation
            assert "score" in citation
            assert 0 <= citation["score"] <= 1

    def test_ask_question_has_follow_up_questions(self, client):
        """Test that response includes follow-up questions."""
        response = client.post(
            "/api/v1/ask",
            json={"question": "Tell me about store operations"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "follow_up_questions" in data
        # Follow-up questions may be None or a list
        if data["follow_up_questions"] is not None:
            assert isinstance(data["follow_up_questions"], list)

    def test_question_api_health(self, client):
        """Test question API health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "question-api"
