"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AnswerResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    ConflictInfo,
    ErrorDetail,
    ErrorResponse,
    FilterInfo,
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


class TestMetaGenerationAttempts:
    """Tests for Meta model generation_attempts field."""

    def test_meta_with_generation_attempts(self):
        """Test meta with generation_attempts field."""
        meta = Meta(
            trace_id="tr_123",
            topk=8,
            retrieval_attempts=2,
            generation_attempts=3,
            latency_ms=500.0,
            verdict=Verdict.PARTIAL,
        )
        assert meta.generation_attempts == 3

    def test_meta_generation_attempts_default(self):
        """Test generation_attempts defaults to 1."""
        meta = Meta(
            trace_id="tr_123",
            topk=5,
            retrieval_attempts=1,
            latency_ms=100.0,
            verdict=Verdict.ANSWERED,
        )
        assert meta.generation_attempts == 1


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_valid_chat_request(self):
        """Test creating a valid chat request."""
        request = ChatRequest(
            question="환불 정책이 어떻게 되나요?",
            store_type="cafe",
            category="refund",
        )
        assert request.question == "환불 정책이 어떻게 되나요?"
        assert request.store_type == "cafe"

    def test_minimal_chat_request(self):
        """Test request with only required fields."""
        request = ChatRequest(question="질문입니다")
        assert request.question == "질문입니다"
        assert request.store_type is None
        assert request.category is None
        assert request.effective_date is None
        assert request.language is None

    def test_chat_request_empty_question_rejected(self):
        """Test that empty questions are rejected."""
        with pytest.raises(ValidationError):
            ChatRequest(question="")

    def test_chat_request_max_length(self):
        """Test question maximum length validation."""
        long_question = "a" * 2001
        with pytest.raises(ValidationError):
            ChatRequest(question=long_question)

    def test_chat_request_with_all_filters(self):
        """Test chat request with all filter fields."""
        request = ChatRequest(
            question="테스트 질문",
            store_type="cafe",
            category="refund",
            effective_date="2024-01-01",
            language="ko",
        )
        assert request.store_type == "cafe"
        assert request.category == "refund"
        assert request.effective_date == "2024-01-01"
        assert request.language == "ko"


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_valid_chat_response(self):
        """Test creating a valid chat response."""
        response = ChatResponse(
            answer="테스트 답변입니다.",
            citations=[
                Citation(
                    doc_id="doc_001",
                    title="테스트 문서",
                    chunk_id="chunk_001",
                    snippet="테스트 스니펫",
                    score=0.9,
                )
            ],
            follow_up_question="더 궁금한 점이 있나요?",
            meta=Meta(
                trace_id="tr_123",
                topk=8,
                retrieval_attempts=1,
                latency_ms=250.0,
                verdict=Verdict.ANSWERED,
            ),
        )
        assert response.answer == "테스트 답변입니다."
        assert len(response.citations) == 1
        assert response.follow_up_question == "더 궁금한 점이 있나요?"

    def test_chat_response_without_optional_fields(self):
        """Test response without optional fields."""
        response = ChatResponse(
            answer="답변",
            meta=Meta(
                trace_id="tr_123",
                topk=5,
                retrieval_attempts=1,
                latency_ms=100.0,
                verdict=Verdict.NOT_FOUND,
            ),
        )
        assert response.citations == []
        assert response.follow_up_question is None

    def test_chat_response_follow_up_is_string(self):
        """Test that follow_up_question is a single string, not a list."""
        response = ChatResponse(
            answer="답변",
            follow_up_question="단일 후속 질문",
            meta=Meta(
                trace_id="tr_123",
                topk=5,
                retrieval_attempts=1,
                latency_ms=100.0,
                verdict=Verdict.ANSWERED,
            ),
        )
        assert isinstance(response.follow_up_question, str)


class TestErrorSchemas:
    """Tests for error-related schemas."""

    def test_error_detail(self):
        """Test ErrorDetail schema."""
        error = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Invalid input",
            trace_id="tr_123",
            details={"field": "question"},
        )
        assert error.code == "VALIDATION_ERROR"
        assert error.trace_id == "tr_123"
        assert error.details == {"field": "question"}

    def test_error_detail_minimal(self):
        """Test ErrorDetail with only required fields."""
        error = ErrorDetail(
            code="INTERNAL_ERROR",
            message="Something went wrong",
        )
        assert error.trace_id is None
        assert error.details is None

    def test_error_response(self):
        """Test ErrorResponse schema."""
        response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="Something went wrong",
            )
        )
        assert response.error.code == "INTERNAL_ERROR"


class TestFilterInfo:
    """Tests for FilterInfo schema."""

    def test_filter_info(self):
        """Test FilterInfo schema."""
        filter_info = FilterInfo(
            store_type="cafe",
            category="refund",
            was_relaxed=True,
            relaxation_message="Category filter was relaxed",
        )
        assert filter_info.store_type == "cafe"
        assert filter_info.was_relaxed is True

    def test_filter_info_defaults(self):
        """Test FilterInfo default values."""
        filter_info = FilterInfo()
        assert filter_info.store_type is None
        assert filter_info.was_relaxed is False
        assert filter_info.relaxation_message is None


class TestConflictInfo:
    """Tests for ConflictInfo schema."""

    def test_conflict_info(self):
        """Test ConflictInfo schema."""
        conflict = ConflictInfo(
            conflicting_citations=["doc_001", "doc_002"],
            resolution_basis="effective_date",
            recommended_citation_id="doc_002",
        )
        assert len(conflict.conflicting_citations) == 2
        assert conflict.resolution_basis == "effective_date"
        assert conflict.recommended_citation_id == "doc_002"

    def test_conflict_info_minimal(self):
        """Test ConflictInfo with only required fields."""
        conflict = ConflictInfo(
            conflicting_citations=["doc_001", "doc_002"],
        )
        assert conflict.resolution_basis is None
        assert conflict.recommended_citation_id is None
