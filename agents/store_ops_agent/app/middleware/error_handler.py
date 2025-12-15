"""Error handling middleware for StoreOps Agent.

Provides centralized error handling with:
- Consistent error response format
- Trace ID propagation for debugging
- Structured logging of errors
"""

import logging
import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.tracing import generate_trace_id
from app.models.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        code: str = "AGENT_ERROR",
        status_code: int = 500,
        details: dict | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ValidationError(AgentError):
    """Exception for validation errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class RetrievalError(AgentError):
    """Exception for retrieval/search errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            code="RETRIEVAL_ERROR",
            status_code=500,
            details=details,
        )


class GenerationError(AgentError):
    """Exception for LLM generation errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            message=message,
            code="GENERATION_ERROR",
            status_code=500,
            details=details,
        )


class RateLimitError(AgentError):
    """Exception for rate limiting."""

    def __init__(self, message: str = "Rate limit exceeded", details: dict | None = None):
        super().__init__(
            message=message,
            code="RATE_LIMIT_ERROR",
            status_code=429,
            details=details,
        )


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for handling exceptions and returning consistent error responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle any exceptions.

        Args:
            request: The incoming request.
            call_next: The next middleware/endpoint in the chain.

        Returns:
            Response from the endpoint or error response.
        """
        # Get or generate trace ID
        trace_id = request.headers.get("X-Trace-ID") or generate_trace_id()
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
            # Add trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id
            return response

        except AgentError as e:
            logger.error(
                f"[{trace_id}] Agent error: {e.message}",
                extra={
                    "trace_id": trace_id,
                    "error_code": e.code,
                    "status_code": e.status_code,
                    "details": e.details,
                    "path": request.url.path,
                    "method": request.method,
                },
            )
            return self._create_error_response(
                trace_id=trace_id,
                code=e.code,
                message=e.message,
                status_code=e.status_code,
                details=e.details,
            )

        except ValueError as e:
            logger.warning(
                f"[{trace_id}] Validation error: {str(e)}",
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method,
                },
            )
            return self._create_error_response(
                trace_id=trace_id,
                code="VALIDATION_ERROR",
                message=str(e),
                status_code=400,
            )

        except Exception as e:
            # Log full traceback for unexpected errors
            logger.exception(
                f"[{trace_id}] Unexpected error: {str(e)}",
                extra={
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method,
                    "traceback": traceback.format_exc(),
                },
            )
            return self._create_error_response(
                trace_id=trace_id,
                code="INTERNAL_ERROR",
                message="An unexpected error occurred. Please try again later.",
                status_code=500,
                details={"error_type": type(e).__name__},
            )

    def _create_error_response(
        self,
        trace_id: str,
        code: str,
        message: str,
        status_code: int,
        details: dict | None = None,
    ) -> JSONResponse:
        """Create a standardized error response.

        Args:
            trace_id: Request trace ID for debugging.
            code: Error code for client handling.
            message: Human-readable error message.
            status_code: HTTP status code.
            details: Additional error details.

        Returns:
            JSONResponse with error information.
        """
        error_response = ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=message,
                trace_id=trace_id,
                details=details,
            )
        )
        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),
            headers={"X-Trace-ID": trace_id},
        )
