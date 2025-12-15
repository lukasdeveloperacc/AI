"""Tracing and metadata collection utilities."""

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from app.models.schemas import Meta, Verdict


def generate_trace_id() -> str:
    """Generate a unique trace ID.

    Returns:
        A unique trace ID string with 'tr_' prefix.
    """
    return f"tr_{uuid.uuid4().hex[:16]}"


@dataclass
class RequestTracer:
    """Tracks request metadata during processing.

    Collects timing, retrieval statistics, and verdict information
    for inclusion in API responses.
    """

    trace_id: str = field(default_factory=generate_trace_id)
    topk: int = 5
    retrieval_attempts: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    verdict: Verdict = Verdict.NOT_FOUND

    def record_retrieval_attempt(self) -> None:
        """Record a retrieval attempt."""
        self.retrieval_attempts += 1

    def set_topk(self, topk: int) -> None:
        """Set the topk value used for retrieval.

        Args:
            topk: Number of top documents to retrieve.
        """
        self.topk = topk

    def set_verdict(self, verdict: Verdict) -> None:
        """Set the answer verdict.

        Args:
            verdict: The confidence level of the answer.
        """
        self.verdict = verdict

    def determine_verdict(
        self,
        citations_count: int,
        min_citations: int = 2,
        has_answer: bool = True,
    ) -> Verdict:
        """Determine verdict based on retrieval results.

        Args:
            citations_count: Number of citations found.
            min_citations: Minimum citations for full answer (default 2).
            has_answer: Whether an answer was generated.

        Returns:
            Appropriate verdict based on results.
        """
        if not has_answer or citations_count == 0:
            verdict = Verdict.NOT_FOUND
        elif citations_count < min_citations:
            verdict = Verdict.PARTIAL
        else:
            verdict = Verdict.ANSWERED

        self.verdict = verdict
        return verdict

    def get_latency_ms(self) -> float:
        """Calculate elapsed time since request start.

        Returns:
            Elapsed time in milliseconds.
        """
        return (time.perf_counter() - self.start_time) * 1000

    def to_meta(self) -> Meta:
        """Convert tracer data to Meta response model.

        Returns:
            Meta object with collected information.
        """
        return Meta(
            trace_id=self.trace_id,
            topk=self.topk,
            retrieval_attempts=max(self.retrieval_attempts, 1),
            latency_ms=round(self.get_latency_ms(), 2),
            verdict=self.verdict,
        )


@contextmanager
def trace_request(topk: int = 5) -> Generator[RequestTracer, None, None]:
    """Context manager for request tracing.

    Args:
        topk: Number of top documents to retrieve.

    Yields:
        RequestTracer instance for the duration of the request.

    Example:
        with trace_request(topk=5) as tracer:
            tracer.record_retrieval_attempt()
            # ... process request ...
            meta = tracer.to_meta()
    """
    tracer = RequestTracer(topk=topk)
    try:
        yield tracer
    finally:
        pass  # Cleanup if needed


class MetricsCollector:
    """Collects and aggregates metrics across requests."""

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._request_count: int = 0
        self._total_latency_ms: float = 0.0
        self._verdict_counts: dict[Verdict, int] = {
            Verdict.ANSWERED: 0,
            Verdict.PARTIAL: 0,
            Verdict.NOT_FOUND: 0,
        }

    def record(self, meta: Meta) -> None:
        """Record metrics from a completed request.

        Args:
            meta: Meta information from the request.
        """
        self._request_count += 1
        self._total_latency_ms += meta.latency_ms
        self._verdict_counts[meta.verdict] += 1

    @property
    def request_count(self) -> int:
        """Get total request count."""
        return self._request_count

    @property
    def average_latency_ms(self) -> float:
        """Get average latency across all requests."""
        if self._request_count == 0:
            return 0.0
        return self._total_latency_ms / self._request_count

    @property
    def success_rate(self) -> float:
        """Get rate of successfully answered requests."""
        if self._request_count == 0:
            return 0.0
        answered = self._verdict_counts[Verdict.ANSWERED]
        return answered / self._request_count

    def get_stats(self) -> dict:
        """Get all collected statistics.

        Returns:
            Dictionary containing all metrics.
        """
        return {
            "request_count": self._request_count,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "verdict_distribution": {
                v.value: count for v, count in self._verdict_counts.items()
            },
        }
