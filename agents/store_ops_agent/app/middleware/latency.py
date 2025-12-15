"""Latency measurement middleware for StoreOps Agent.

Provides:
- Request timing measurement
- Latency logging with structured data
- Response header injection with timing info
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Paths to exclude from latency logging (e.g., health checks)
EXCLUDED_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}


class LatencyMiddleware(BaseHTTPMiddleware):
    """Middleware for measuring and logging request latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Measure request processing time and add to response headers.

        Args:
            request: The incoming request.
            call_next: The next middleware/endpoint in the chain.

        Returns:
            Response with latency headers added.
        """
        start_time = time.perf_counter()

        # Process the request
        response = await call_next(request)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Add latency to response headers
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"

        # Log latency for non-excluded paths
        if request.url.path not in EXCLUDED_PATHS:
            trace_id = getattr(request.state, "trace_id", "unknown")

            log_data = {
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "latency_ms": round(latency_ms, 2),
                "status_code": response.status_code,
            }

            # Log at different levels based on latency
            if latency_ms > 2000:  # > 2 seconds (p95 requirement)
                logger.warning(
                    f"[{trace_id}] Slow request: {request.method} {request.url.path} "
                    f"took {latency_ms:.2f}ms (exceeds p95 target)",
                    extra=log_data,
                )
            elif latency_ms > 1000:  # > 1 second
                logger.info(
                    f"[{trace_id}] Request: {request.method} {request.url.path} "
                    f"took {latency_ms:.2f}ms",
                    extra=log_data,
                )
            else:
                logger.debug(
                    f"[{trace_id}] Request: {request.method} {request.url.path} "
                    f"took {latency_ms:.2f}ms",
                    extra=log_data,
                )

        return response


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for adding request context (client info, etc.)."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request context for logging.

        Args:
            request: The incoming request.
            call_next: The next middleware/endpoint in the chain.

        Returns:
            Response from the endpoint.
        """
        # Extract client info
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        # Store in request state for access in endpoints
        request.state.client_host = client_host
        request.state.user_agent = user_agent

        return await call_next(request)
