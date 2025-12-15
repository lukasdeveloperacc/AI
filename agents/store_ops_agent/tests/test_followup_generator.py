"""Tests for follow-up question generator."""

import pytest

from app.core.followup_generator import FollowUpGenerator
from app.models.schemas import Citation, Verdict


@pytest.fixture
def generator() -> FollowUpGenerator:
    """Create a follow-up generator instance."""
    return FollowUpGenerator(max_questions=3)


@pytest.fixture
def sample_citations() -> list[Citation]:
    """Create sample citations for testing."""
    return [
        Citation(
            doc_id="doc_001",
            title="매장 운영 매뉴얼",
            chunk_id="chunk_001",
            snippet="영업시간 관련 정보",
            score=0.85,
        ),
        Citation(
            doc_id="doc_002",
            title="직원 근무 규정",
            chunk_id="chunk_002",
            snippet="근무 시간 정보",
            score=0.55,
        ),
    ]


class TestFollowUpGenerator:
    """Tests for FollowUpGenerator class."""

    def test_generates_questions_for_not_found(
        self,
        generator: FollowUpGenerator,
    ) -> None:
        """Test generating questions for NOT_FOUND verdict."""
        questions = generator.generate(
            verdict=Verdict.NOT_FOUND,
            question="영업시간이 어떻게 되나요?",
        )

        assert len(questions) > 0
        assert len(questions) <= 3

    def test_generates_questions_for_insufficient(
        self,
        generator: FollowUpGenerator,
    ) -> None:
        """Test generating questions for INSUFFICIENT verdict."""
        questions = generator.generate(
            verdict=Verdict.INSUFFICIENT,
            question="환불 규정이 뭔가요?",
        )

        assert len(questions) > 0
        assert len(questions) <= 3

    def test_generates_questions_for_conflict(
        self,
        generator: FollowUpGenerator,
        sample_citations: list[Citation],
    ) -> None:
        """Test generating questions for CONFLICT verdict."""
        questions = generator.generate(
            verdict=Verdict.CONFLICT,
            question="영업시간이 어떻게 되나요?",
            citations=sample_citations,
        )

        assert len(questions) > 0
        assert len(questions) <= 3

    def test_respects_max_questions_limit(
        self,
        generator: FollowUpGenerator,
    ) -> None:
        """Test that generator respects max_questions limit."""
        questions = generator.generate(
            verdict=Verdict.NOT_FOUND,
            question="영업시간이 어떻게 되나요?",
        )

        assert len(questions) <= generator.max_questions

    def test_context_question_for_time_related(
        self,
        generator: FollowUpGenerator,
    ) -> None:
        """Test context-specific question for time-related queries."""
        questions = generator.generate(
            verdict=Verdict.INSUFFICIENT,
            question="언제 문을 여나요?",
        )

        # Should include a time-specific follow-up
        assert any("요일" in q or "기간" in q for q in questions)

    def test_context_question_for_method_related(
        self,
        generator: FollowUpGenerator,
    ) -> None:
        """Test context-specific question for method-related queries."""
        questions = generator.generate(
            verdict=Verdict.INSUFFICIENT,
            question="어떻게 신청하나요?",
        )

        # Should include a method-specific follow-up
        assert any("상황" in q for q in questions)

    def test_citation_based_questions_for_conflict(
        self,
    ) -> None:
        """Test that citation-based questions are generated for conflicts."""
        generator = FollowUpGenerator(max_questions=5)
        citations = [
            Citation(
                doc_id="doc_001",
                title="매뉴얼 2024",
                chunk_id="chunk_001",
                snippet="정보 1",
                score=0.85,
            ),
            Citation(
                doc_id="doc_002",
                title="매뉴얼 2023",
                chunk_id="chunk_002",
                snippet="정보 2",
                score=0.83,
            ),
        ]

        questions = generator.generate(
            verdict=Verdict.CONFLICT,
            question="영업시간이 어떻게 되나요?",
            citations=citations,
        )

        # Should mention the conflicting documents
        assert any("매뉴얼" in q for q in questions)

    def test_no_duplicate_questions(
        self,
        generator: FollowUpGenerator,
        sample_citations: list[Citation],
    ) -> None:
        """Test that generated questions are unique."""
        questions = generator.generate(
            verdict=Verdict.INSUFFICIENT,
            question="영업시간이 어떻게 되나요?",
            citations=sample_citations,
        )

        # All questions should be unique
        assert len(questions) == len(set(questions))


class TestKeyTermExtraction:
    """Tests for key term extraction."""

    def test_extracts_meaningful_terms(self) -> None:
        """Test extraction of meaningful terms from question."""
        result = FollowUpGenerator._extract_key_term("영업시간이 어떻게 되나요?")

        # Should extract meaningful terms
        assert "영업시간" in result or "영업" in result

    def test_handles_empty_question(self) -> None:
        """Test handling of empty question."""
        result = FollowUpGenerator._extract_key_term("")

        assert result == ""

    def test_handles_single_word(self) -> None:
        """Test handling of single word question."""
        result = FollowUpGenerator._extract_key_term("환불")

        assert "환불" in result

    def test_removes_stop_words(self) -> None:
        """Test removal of common stop words."""
        result = FollowUpGenerator._extract_key_term("매장의 운영 방법을 알려주세요")

        # Should remove stop words and keep meaningful terms
        assert "알려주세요" not in result


class TestPartialVerdictFollowUps:
    """Tests for follow-up questions in PARTIAL verdict scenario."""

    def test_generates_partial_followups(self) -> None:
        """Test that partial verdict generates appropriate follow-ups."""
        generator = FollowUpGenerator(max_questions=3)

        questions = generator.generate(
            verdict=Verdict.PARTIAL,
            question="환불 규정이 뭔가요?",
        )

        # Should have follow-up questions for partial answers
        assert len(questions) > 0
        assert any("추가" in q or "다른" in q for q in questions)
