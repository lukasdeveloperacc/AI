"""Tests for middleware components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.middleware.error_handler import (
    AgentError,
    ErrorHandlerMiddleware,
    GenerationError,
    RateLimitError,
    RetrievalError,
    ValidationError,
)
from app.middleware.latency import LatencyMiddleware


@pytest.fixture
def test_app():
    """Create a test FastAPI app with middleware."""
    app = FastAPI()
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(LatencyMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}

    @app.get("/error")
    async def error_endpoint():
        raise Exception("Test error")

    @app.get("/agent-error")
    async def agent_error_endpoint():
        raise AgentError(
            message="Agent failed",
            code="AGENT_FAILURE",
            status_code=500,
        )

    @app.get("/validation-error")
    async def validation_error_endpoint():
        raise ValidationError(
            message="Invalid input",
            details={"field": "question"},
        )

    @app.get("/value-error")
    async def value_error_endpoint():
        raise ValueError("Invalid value")

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestErrorHandlerMiddleware:
    """Tests for ErrorHandlerMiddleware."""

    def test_successful_request(self, client):
        """Test that successful requests pass through."""
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"message": "success"}

    def test_trace_id_in_response(self, client):
        """Test that trace ID is added to response headers."""
        response = client.get("/test")
        assert "x-trace-id" in response.headers

    def test_unexpected_error_handling(self, client):
        """Test handling of unexpected exceptions."""
        response = client.get("/error")
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert "trace_id" in data["error"]

    def test_agent_error_handling(self, client):
        """Test handling of AgentError exceptions."""
        response = client.get("/agent-error")
        assert response.status_code == 500

        data = response.json()
        assert data["error"]["code"] == "AGENT_FAILURE"
        assert data["error"]["message"] == "Agent failed"

    def test_validation_error_handling(self, client):
        """Test handling of ValidationError exceptions."""
        response = client.get("/validation-error")
        assert response.status_code == 400

        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert data["error"]["details"] == {"field": "question"}

    def test_value_error_handling(self, client):
        """Test handling of ValueError exceptions."""
        response = client.get("/value-error")
        assert response.status_code == 400

        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_custom_trace_id_header(self, client):
        """Test that custom X-Trace-ID header is respected."""
        custom_trace_id = "tr_custom123"
        response = client.get(
            "/test",
            headers={"X-Trace-ID": custom_trace_id},
        )
        assert response.headers["x-trace-id"] == custom_trace_id


class TestLatencyMiddleware:
    """Tests for LatencyMiddleware."""

    def test_latency_header_added(self, client):
        """Test that X-Response-Time-Ms header is added."""
        response = client.get("/test")
        assert "x-response-time-ms" in response.headers

        # Latency should be a valid float
        latency = float(response.headers["x-response-time-ms"])
        assert latency >= 0


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_agent_error(self):
        """Test AgentError exception."""
        error = AgentError(
            message="Test error",
            code="TEST_CODE",
            status_code=503,
            details={"key": "value"},
        )
        assert error.message == "Test error"
        assert error.code == "TEST_CODE"
        assert error.status_code == 503
        assert error.details == {"key": "value"}

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError(
            message="Invalid input",
            details={"field": "name"},
        )
        assert error.code == "VALIDATION_ERROR"
        assert error.status_code == 400

    def test_retrieval_error(self):
        """Test RetrievalError exception."""
        error = RetrievalError(message="Search failed")
        assert error.code == "RETRIEVAL_ERROR"
        assert error.status_code == 500

    def test_generation_error(self):
        """Test GenerationError exception."""
        error = GenerationError(message="LLM failed")
        assert error.code == "GENERATION_ERROR"
        assert error.status_code == 500

    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        error = RateLimitError()
        assert error.code == "RATE_LIMIT_ERROR"
        assert error.status_code == 429
        assert error.message == "Rate limit exceeded"
