"""Tests for tracing utilities."""

import time

import pytest

from app.core.tracing import (
    MetricsCollector,
    RequestTracer,
    generate_trace_id,
    trace_request,
)
from app.models.schemas import Meta, Verdict


class TestGenerateTraceId:
    """Tests for generate_trace_id function."""

    def test_generates_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_format(self):
        """Test trace ID format."""
        trace_id = generate_trace_id()
        assert trace_id.startswith("tr_")
        assert len(trace_id) == 19  # tr_ + 16 hex chars


class TestRequestTracer:
    """Tests for RequestTracer class."""

    def test_default_values(self):
        """Test default tracer values."""
        tracer = RequestTracer()
        assert tracer.topk == 5
        assert tracer.retrieval_attempts == 0
        assert tracer.verdict == Verdict.NOT_FOUND
        assert tracer.trace_id.startswith("tr_")

    def test_record_retrieval_attempt(self):
        """Test recording retrieval attempts."""
        tracer = RequestTracer()
        assert tracer.retrieval_attempts == 0
        tracer.record_retrieval_attempt()
        assert tracer.retrieval_attempts == 1
        tracer.record_retrieval_attempt()
        assert tracer.retrieval_attempts == 2

    def test_set_topk(self):
        """Test setting topk value."""
        tracer = RequestTracer()
        tracer.set_topk(10)
        assert tracer.topk == 10

    def test_set_verdict(self):
        """Test setting verdict."""
        tracer = RequestTracer()
        tracer.set_verdict(Verdict.ANSWERED)
        assert tracer.verdict == Verdict.ANSWERED

    def test_determine_verdict_answered(self):
        """Test verdict determination with sufficient citations."""
        tracer = RequestTracer()
        verdict = tracer.determine_verdict(citations_count=3, min_citations=2)
        assert verdict == Verdict.ANSWERED
        assert tracer.verdict == Verdict.ANSWERED

    def test_determine_verdict_partial(self):
        """Test verdict determination with insufficient citations."""
        tracer = RequestTracer()
        verdict = tracer.determine_verdict(citations_count=1, min_citations=2)
        assert verdict == Verdict.PARTIAL
        assert tracer.verdict == Verdict.PARTIAL

    def test_determine_verdict_not_found(self):
        """Test verdict determination with no citations."""
        tracer = RequestTracer()
        verdict = tracer.determine_verdict(citations_count=0)
        assert verdict == Verdict.NOT_FOUND

    def test_determine_verdict_no_answer(self):
        """Test verdict determination when no answer generated."""
        tracer = RequestTracer()
        verdict = tracer.determine_verdict(citations_count=5, has_answer=False)
        assert verdict == Verdict.NOT_FOUND

    def test_get_latency_ms(self):
        """Test latency calculation."""
        tracer = RequestTracer()
        time.sleep(0.01)  # Sleep 10ms
        latency = tracer.get_latency_ms()
        assert latency >= 10  # At least 10ms

    def test_to_meta(self):
        """Test conversion to Meta model."""
        tracer = RequestTracer(topk=10)
        tracer.record_retrieval_attempt()
        tracer.set_verdict(Verdict.ANSWERED)

        meta = tracer.to_meta()
        assert isinstance(meta, Meta)
        assert meta.topk == 10
        assert meta.retrieval_attempts == 1
        assert meta.verdict == Verdict.ANSWERED
        assert meta.latency_ms >= 0

    def test_to_meta_minimum_retrieval_attempts(self):
        """Test that retrieval_attempts is at least 1 in meta."""
        tracer = RequestTracer()
        # No retrieval attempts recorded
        meta = tracer.to_meta()
        assert meta.retrieval_attempts == 1


class TestTraceRequestContextManager:
    """Tests for trace_request context manager."""

    def test_context_manager(self):
        """Test basic context manager usage."""
        with trace_request(topk=7) as tracer:
            assert tracer.topk == 7
            tracer.record_retrieval_attempt()
            tracer.set_verdict(Verdict.ANSWERED)

        # Tracer should have recorded data
        assert tracer.retrieval_attempts == 1
        assert tracer.verdict == Verdict.ANSWERED

    def test_context_manager_measures_time(self):
        """Test that context manager measures elapsed time."""
        with trace_request() as tracer:
            time.sleep(0.01)
            meta = tracer.to_meta()

        assert meta.latency_ms >= 10


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initial_state(self):
        """Test initial collector state."""
        collector = MetricsCollector()
        assert collector.request_count == 0
        assert collector.average_latency_ms == 0.0
        assert collector.success_rate == 0.0

    def test_record_single_request(self):
        """Test recording a single request."""
        collector = MetricsCollector()
        meta = Meta(
            trace_id="tr_test",
            topk=5,
            retrieval_attempts=1,
            latency_ms=100.0,
            verdict=Verdict.ANSWERED,
        )
        collector.record(meta)
        assert collector.request_count == 1
        assert collector.average_latency_ms == 100.0
        assert collector.success_rate == 1.0

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        collector = MetricsCollector()

        metas = [
            Meta(
                trace_id="tr_1",
                topk=5,
                retrieval_attempts=1,
                latency_ms=100.0,
                verdict=Verdict.ANSWERED,
            ),
            Meta(
                trace_id="tr_2",
                topk=5,
                retrieval_attempts=1,
                latency_ms=200.0,
                verdict=Verdict.PARTIAL,
            ),
            Meta(
                trace_id="tr_3",
                topk=5,
                retrieval_attempts=2,
                latency_ms=150.0,
                verdict=Verdict.NOT_FOUND,
            ),
        ]

        for meta in metas:
            collector.record(meta)

        assert collector.request_count == 3
        assert collector.average_latency_ms == 150.0
        assert collector.success_rate == pytest.approx(1 / 3, rel=0.01)

    def test_get_stats(self):
        """Test getting all statistics."""
        collector = MetricsCollector()
        meta = Meta(
            trace_id="tr_test",
            topk=5,
            retrieval_attempts=1,
            latency_ms=100.0,
            verdict=Verdict.ANSWERED,
        )
        collector.record(meta)

        stats = collector.get_stats()
        assert "request_count" in stats
        assert "average_latency_ms" in stats
        assert "success_rate" in stats
        assert "verdict_distribution" in stats
        assert stats["verdict_distribution"]["answered"] == 1
