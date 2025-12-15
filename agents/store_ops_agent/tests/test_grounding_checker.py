"""Tests for grounding checker and fail-safe policy."""

import pytest

from app.core.grounding_checker import (
    FailSafePolicy,
    GroundingChecker,
    GroundingResult,
)
from app.models.schemas import Citation, Verdict


@pytest.fixture
def grounding_checker() -> GroundingChecker:
    """Create a grounding checker instance."""
    return GroundingChecker(
        min_citations=1,
        min_score_threshold=0.6,
        conflict_score_diff_threshold=0.15,
    )


@pytest.fixture
def high_score_citation() -> Citation:
    """Create a high-score citation."""
    return Citation(
        doc_id="doc_001",
        title="매장 운영 매뉴얼",
        chunk_id="chunk_001",
        snippet="영업시간은 오전 9시부터 오후 10시까지입니다.",
        score=0.92,
        effective_date="2024-01-01",
        version="v2.0",
    )


@pytest.fixture
def low_score_citation() -> Citation:
    """Create a low-score citation."""
    return Citation(
        doc_id="doc_002",
        title="직원 근무 규정",
        chunk_id="chunk_002",
        snippet="근무 시간 관련 정보입니다.",
        score=0.45,
    )


@pytest.fixture
def conflicting_citation() -> Citation:
    """Create a citation that conflicts with high_score_citation."""
    return Citation(
        doc_id="doc_003",
        title="이전 매뉴얼",
        chunk_id="chunk_003",
        snippet="영업시간은 오전 10시부터 오후 9시까지입니다.",
        score=0.90,
        effective_date="2023-06-01",
        version="v1.0",
    )


class TestGroundingChecker:
    """Tests for GroundingChecker class."""

    def test_no_citations_returns_not_found(
        self,
        grounding_checker: GroundingChecker,
    ) -> None:
        """Test that empty citations returns NOT_FOUND verdict."""
        result = grounding_checker.check([], "영업시간이 어떻게 되나요?")

        assert result.verdict == Verdict.NOT_FOUND
        assert result.is_sufficient is False
        assert result.withheld_reason is not None
        assert "검색 결과가 없습니다" in result.withheld_reason

    def test_low_score_citations_returns_insufficient(
        self,
        grounding_checker: GroundingChecker,
        low_score_citation: Citation,
    ) -> None:
        """Test that only low-score citations returns INSUFFICIENT verdict."""
        result = grounding_checker.check([low_score_citation], "영업시간이 어떻게 되나요?")

        assert result.verdict == Verdict.INSUFFICIENT
        assert result.is_sufficient is False

    def test_high_score_citation_returns_answered(
        self,
        grounding_checker: GroundingChecker,
        high_score_citation: Citation,
    ) -> None:
        """Test that high-score citation returns ANSWERED verdict."""
        result = grounding_checker.check([high_score_citation], "영업시간이 어떻게 되나요?")

        assert result.verdict == Verdict.ANSWERED
        assert result.is_sufficient is True

    def test_conflicting_citations_returns_conflict(
        self,
        grounding_checker: GroundingChecker,
        high_score_citation: Citation,
        conflicting_citation: Citation,
    ) -> None:
        """Test that conflicting citations returns CONFLICT verdict."""
        citations = [high_score_citation, conflicting_citation]
        result = grounding_checker.check(citations, "영업시간이 어떻게 되나요?")

        assert result.verdict == Verdict.CONFLICT
        assert result.is_sufficient is False
        assert result.conflict_info is not None
        assert len(result.conflict_info.conflicting_citations) >= 2
        assert result.conflict_info.resolution_basis == "effective_date"
        assert result.conflict_info.recommended_citation_id == high_score_citation.doc_id

    def test_conflict_resolution_by_version(
        self,
        grounding_checker: GroundingChecker,
    ) -> None:
        """Test conflict resolution by version when dates are not available."""
        citation1 = Citation(
            doc_id="doc_a",
            title="문서 A",
            chunk_id="chunk_a",
            snippet="정보 A",
            score=0.85,
            version="v2.0",
        )
        citation2 = Citation(
            doc_id="doc_b",
            title="문서 B",
            chunk_id="chunk_b",
            snippet="정보 B",
            score=0.84,
            version="v1.0",
        )

        result = grounding_checker.check([citation1, citation2], "질문")

        assert result.verdict == Verdict.CONFLICT
        assert result.conflict_info is not None
        assert result.conflict_info.resolution_basis == "version"
        assert result.conflict_info.recommended_citation_id == citation1.doc_id

    def test_partial_verdict_for_medium_score_citations(
        self,
        grounding_checker: GroundingChecker,
    ) -> None:
        """Test PARTIAL verdict when citations are valid but not high quality."""
        citation = Citation(
            doc_id="doc_001",
            title="문서",
            chunk_id="chunk_001",
            snippet="관련 정보",
            score=0.65,  # Above threshold but below 0.8
        )

        result = grounding_checker.check([citation], "질문")

        assert result.verdict == Verdict.PARTIAL
        assert result.is_sufficient is True

    def test_no_conflict_for_same_document(
        self,
        grounding_checker: GroundingChecker,
    ) -> None:
        """Test that citations from same document don't conflict."""
        citation1 = Citation(
            doc_id="doc_001",
            title="문서",
            chunk_id="chunk_001",
            snippet="정보 1",
            score=0.90,
            effective_date="2024-01-01",
        )
        citation2 = Citation(
            doc_id="doc_001",  # Same doc_id
            title="문서",
            chunk_id="chunk_002",
            snippet="정보 2",
            score=0.88,
            effective_date="2023-01-01",
        )

        result = grounding_checker.check([citation1, citation2], "질문")

        # Should not detect conflict for same document
        assert result.verdict != Verdict.CONFLICT


class TestFailSafePolicy:
    """Tests for FailSafePolicy class."""

    def test_should_withhold_for_not_found(self) -> None:
        """Test that answer is withheld for NOT_FOUND verdict."""
        result = GroundingResult(
            verdict=Verdict.NOT_FOUND,
            is_sufficient=False,
        )

        assert FailSafePolicy.should_withhold_answer(result) is True

    def test_should_withhold_for_insufficient(self) -> None:
        """Test that answer is withheld for INSUFFICIENT verdict."""
        result = GroundingResult(
            verdict=Verdict.INSUFFICIENT,
            is_sufficient=False,
        )

        assert FailSafePolicy.should_withhold_answer(result) is True

    def test_should_not_withhold_for_conflict(self) -> None:
        """Test that answer is NOT withheld for CONFLICT (we provide cautious answer)."""
        result = GroundingResult(
            verdict=Verdict.CONFLICT,
            is_sufficient=False,
        )

        assert FailSafePolicy.should_withhold_answer(result) is False

    def test_should_not_withhold_for_answered(self) -> None:
        """Test that answer is NOT withheld for ANSWERED verdict."""
        result = GroundingResult(
            verdict=Verdict.ANSWERED,
            is_sufficient=True,
        )

        assert FailSafePolicy.should_withhold_answer(result) is False

    def test_get_withheld_message_for_not_found(self) -> None:
        """Test withheld message for NOT_FOUND verdict."""
        result = GroundingResult(
            verdict=Verdict.NOT_FOUND,
            is_sufficient=False,
        )

        message = FailSafePolicy.get_withheld_message(result)

        assert "관련 문서를 찾을 수 없습니다" in message

    def test_get_withheld_message_for_insufficient(self) -> None:
        """Test withheld message for INSUFFICIENT verdict."""
        result = GroundingResult(
            verdict=Verdict.INSUFFICIENT,
            is_sufficient=False,
        )

        message = FailSafePolicy.get_withheld_message(result)

        assert "확답" in message or "확인" in message

    def test_get_withheld_message_uses_custom_reason(self) -> None:
        """Test that custom withheld_reason is used when provided."""
        custom_reason = "커스텀 메시지입니다."
        result = GroundingResult(
            verdict=Verdict.INSUFFICIENT,
            is_sufficient=False,
            withheld_reason=custom_reason,
        )

        message = FailSafePolicy.get_withheld_message(result)

        assert message == custom_reason


class TestGroundingCheckerDateParsing:
    """Tests for date parsing in GroundingChecker."""

    def test_parse_date_iso_format(self) -> None:
        """Test parsing ISO date format."""
        result = GroundingChecker._parse_date("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_slash_format(self) -> None:
        """Test parsing slash date format."""
        result = GroundingChecker._parse_date("2024/01/15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_iso_datetime(self) -> None:
        """Test parsing ISO datetime format."""
        result = GroundingChecker._parse_date("2024-01-15T10:30:00")
        assert result.year == 2024

    def test_parse_date_returns_min_for_invalid(self) -> None:
        """Test that invalid date returns datetime.min."""
        from datetime import datetime

        result = GroundingChecker._parse_date("invalid-date")
        assert result == datetime.min

    def test_parse_date_returns_min_for_none(self) -> None:
        """Test that None returns datetime.min."""
        from datetime import datetime

        result = GroundingChecker._parse_date(None)
        assert result == datetime.min
