"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AnswerResponse,
    Citation,
    Meta,
    QuestionRequest,
    Verdict,
)


class TestCitation:
    """Tests for Citation model."""

    def test_valid_citation(self):
        """Test creating a valid citation."""
        citation = Citation(
            doc_id="doc_001",
            title="Test Document",
            chunk_id="chunk_001",
            snippet="This is a test snippet.",
            score=0.85,
        )
        assert citation.doc_id == "doc_001"
        assert citation.title == "Test Document"
        assert citation.score == 0.85

    def test_citation_score_bounds(self):
        """Test that score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Citation(
                doc_id="doc_001",
                title="Test",
                chunk_id="chunk_001",
                snippet="Test",
                score=1.5,
            )

        with pytest.raises(ValidationError):
            Citation(
                doc_id="doc_001",
                title="Test",
                chunk_id="chunk_001",
                snippet="Test",
                score=-0.1,
            )

    def test_citation_boundary_scores(self):
        """Test boundary values for score."""
        citation_zero = Citation(
            doc_id="doc_001",
            title="Test",
            chunk_id="chunk_001",
            snippet="Test",
            score=0.0,
        )
        assert citation_zero.score == 0.0

        citation_one = Citation(
            doc_id="doc_001",
            title="Test",
            chunk_id="chunk_001",
            snippet="Test",
            score=1.0,
        )
        assert citation_one.score == 1.0


class TestMeta:
    """Tests for Meta model."""

    def test_valid_meta(self):
        """Test creating valid meta."""
        meta = Meta(
            trace_id="tr_abc123",
            topk=5,
            retrieval_attempts=1,
            latency_ms=100.5,
            verdict=Verdict.ANSWERED,
        )
        assert meta.trace_id == "tr_abc123"
        assert meta.topk == 5
        assert meta.verdict == Verdict.ANSWERED

    def test_meta_verdict_values(self):
        """Test all verdict values are valid."""
        for verdict in Verdict:
            meta = Meta(
                trace_id="tr_test",
                topk=5,
                retrieval_attempts=1,
                latency_ms=100.0,
                verdict=verdict,
            )
            assert meta.verdict == verdict

    def test_meta_invalid_topk(self):
        """Test that topk must be at least 1."""
        with pytest.raises(ValidationError):
            Meta(
                trace_id="tr_test",
                topk=0,
                retrieval_attempts=1,
                latency_ms=100.0,
                verdict=Verdict.ANSWERED,
            )


class TestQuestionRequest:
    """Tests for QuestionRequest model."""

    def test_valid_question(self):
        """Test creating a valid question request."""
        request = QuestionRequest(question="What are the store hours?")
        assert request.question == "What are the store hours?"
        assert request.topk == 5  # default value

    def test_question_with_custom_topk(self):
        """Test question with custom topk."""
        request = QuestionRequest(question="Test question", topk=10)
        assert request.topk == 10

    def test_empty_question_rejected(self):
        """Test that empty questions are rejected."""
        with pytest.raises(ValidationError):
            QuestionRequest(question="")

    def test_topk_bounds(self):
        """Test topk must be within bounds."""
        with pytest.raises(ValidationError):
            QuestionRequest(question="Test", topk=0)

        with pytest.raises(ValidationError):
            QuestionRequest(question="Test", topk=25)


class TestAnswerResponse:
    """Tests for AnswerResponse model."""

    def test_valid_answer_response(self):
        """Test creating a valid answer response."""
        citation = Citation(
            doc_id="doc_001",
            title="Test Doc",
            chunk_id="chunk_001",
            snippet="Test snippet",
            score=0.9,
        )
        meta = Meta(
            trace_id="tr_test",
            topk=5,
            retrieval_attempts=1,
            latency_ms=100.0,
            verdict=Verdict.ANSWERED,
        )
        response = AnswerResponse(
            answer="This is the answer.",
            citations=[citation],
            meta=meta,
        )
        assert response.answer == "This is the answer."
        assert len(response.citations) == 1
        assert response.follow_up_questions is None

    def test_answer_with_follow_up_questions(self):
        """Test answer with follow-up questions."""
        meta = Meta(
            trace_id="tr_test",
            topk=5,
            retrieval_attempts=1,
            latency_ms=100.0,
            verdict=Verdict.ANSWERED,
        )
        response = AnswerResponse(
            answer="This is the answer.",
            citations=[],
            follow_up_questions=["Question 1?", "Question 2?"],
            meta=meta,
        )
        assert response.follow_up_questions is not None
        assert len(response.follow_up_questions) == 2

    def test_answer_empty_citations_allowed(self):
        """Test that empty citations list is allowed."""
        meta = Meta(
            trace_id="tr_test",
            topk=5,
            retrieval_attempts=1,
            latency_ms=100.0,
            verdict=Verdict.NOT_FOUND,
        )
        response = AnswerResponse(
            answer="No relevant information found.",
            citations=[],
            meta=meta,
        )
        assert len(response.citations) == 0
