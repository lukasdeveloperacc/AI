"""Grounding checker for evidence validation and fail-safe logic.

This module implements the fail-safe policy that prevents definitive answers
when evidence is insufficient or conflicting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.models.schemas import Citation, ConflictInfo, Verdict


@dataclass
class GroundingResult:
    """Result of grounding check."""

    verdict: Verdict
    is_sufficient: bool
    conflict_info: Optional[ConflictInfo] = None
    recommended_citation: Optional[Citation] = None
    withheld_reason: Optional[str] = None


class GroundingChecker:
    """Validates evidence sufficiency and detects conflicts.

    Implements the fail-safe policy:
    - Prevents definitive answers when evidence is insufficient
    - Detects and handles conflicting information
    - Recommends the most recent/authoritative source when conflicts exist
    """

    def __init__(
        self,
        min_citations: int = 1,
        min_score_threshold: float = 0.6,
        conflict_score_diff_threshold: float = 0.15,
    ) -> None:
        """Initialize the grounding checker.

        Args:
            min_citations: Minimum number of citations required for a definitive answer.
            min_score_threshold: Minimum relevance score for citations to be considered valid.
            conflict_score_diff_threshold: Score difference threshold below which
                citations may be considered conflicting.
        """
        self.min_citations = min_citations
        self.min_score_threshold = min_score_threshold
        self.conflict_score_diff_threshold = conflict_score_diff_threshold

    def check(
        self,
        citations: list[Citation],
        question: str,
    ) -> GroundingResult:
        """Perform grounding check on citations.

        Args:
            citations: List of retrieved citations.
            question: The user's question for context.

        Returns:
            GroundingResult with verdict and related information.
        """
        # Case 1: No citations found
        if not citations:
            return GroundingResult(
                verdict=Verdict.NOT_FOUND,
                is_sufficient=False,
                withheld_reason="검색 결과가 없습니다. 다른 키워드로 질문해 주세요.",
            )

        # Filter citations by minimum score
        valid_citations = [c for c in citations if c.score >= self.min_score_threshold]

        # Case 2: No citations meet the threshold
        if not valid_citations:
            return GroundingResult(
                verdict=Verdict.INSUFFICIENT,
                is_sufficient=False,
                withheld_reason="관련 문서가 있으나 신뢰도가 충분하지 않습니다.",
            )

        # Case 3: Check for conflicts
        conflict_result = self._detect_conflicts(valid_citations)
        if conflict_result:
            return conflict_result

        # Case 4: Insufficient citations
        if len(valid_citations) < self.min_citations:
            return GroundingResult(
                verdict=Verdict.INSUFFICIENT,
                is_sufficient=False,
                withheld_reason="현재 문서로는 확답을 드리기 어렵습니다.",
            )

        # Case 5: Partial answer (some but not enough high-quality citations)
        high_score_citations = [c for c in valid_citations if c.score >= 0.8]
        if not high_score_citations and len(valid_citations) >= self.min_citations:
            return GroundingResult(
                verdict=Verdict.PARTIAL,
                is_sufficient=True,  # Can still provide answer, but partial
            )

        # Case 6: Sufficient evidence
        return GroundingResult(
            verdict=Verdict.ANSWERED,
            is_sufficient=True,
        )

    def _detect_conflicts(
        self,
        citations: list[Citation],
    ) -> Optional[GroundingResult]:
        """Detect conflicting information in citations.

        Conflicts are detected when citations have similar scores but may
        contain contradictory information. In real implementation, this would
        use semantic analysis.

        Args:
            citations: List of valid citations to check.

        Returns:
            GroundingResult if conflict detected, None otherwise.
        """
        if len(citations) < 2:
            return None

        # Sort by score descending
        sorted_citations = sorted(citations, key=lambda c: c.score, reverse=True)

        # Check for potential conflicts between top citations
        # In a real implementation, this would compare semantic content
        top_citation = sorted_citations[0]

        potential_conflicts = []
        for citation in sorted_citations[1:]:
            score_diff = top_citation.score - citation.score
            if score_diff <= self.conflict_score_diff_threshold:
                # Citations with similar scores might conflict
                # Check if they have different effective dates or versions
                if self._may_conflict(top_citation, citation):
                    potential_conflicts.append(citation)

        if potential_conflicts:
            conflicting_ids = [top_citation.doc_id] + [c.doc_id for c in potential_conflicts]

            # Resolve by effective_date/version
            recommended, resolution_basis = self._resolve_conflict(
                [top_citation] + potential_conflicts
            )

            return GroundingResult(
                verdict=Verdict.CONFLICT,
                is_sufficient=False,
                conflict_info=ConflictInfo(
                    conflicting_citations=conflicting_ids,
                    resolution_basis=resolution_basis,
                    recommended_citation_id=recommended.doc_id if recommended else None,
                ),
                recommended_citation=recommended,
                withheld_reason=(
                    "문서 간 상충되는 정보가 발견되었습니다. "
                    f"{resolution_basis} 기준으로 최신 정보를 안내해 드립니다."
                    if resolution_basis
                    else "문서 간 상충되는 정보가 발견되었습니다."
                ),
            )

        return None

    def _may_conflict(
        self,
        citation1: Citation,
        citation2: Citation,
    ) -> bool:
        """Check if two citations may contain conflicting information.

        This is a simplified check based on different document IDs and dates.
        In a real implementation, this would use semantic comparison.

        Args:
            citation1: First citation.
            citation2: Second citation.

        Returns:
            True if citations may conflict.
        """
        # Different documents with different dates/versions may conflict
        if citation1.doc_id == citation2.doc_id:
            return False

        # Check if they have different effective dates or versions
        has_different_dates = (
            citation1.effective_date != citation2.effective_date
            and citation1.effective_date is not None
            and citation2.effective_date is not None
        )

        has_different_versions = (
            citation1.version != citation2.version
            and citation1.version is not None
            and citation2.version is not None
        )

        return has_different_dates or has_different_versions

    def _resolve_conflict(
        self,
        conflicting_citations: list[Citation],
    ) -> tuple[Optional[Citation], Optional[str]]:
        """Resolve conflict by selecting the most authoritative citation.

        Priority:
        1. Most recent effective_date
        2. Most recent version
        3. Highest score (fallback)

        Args:
            conflicting_citations: List of conflicting citations.

        Returns:
            Tuple of (recommended citation, resolution basis).
        """
        if not conflicting_citations:
            return None, None

        # Try to resolve by effective_date
        citations_with_dates = [
            c for c in conflicting_citations if c.effective_date is not None
        ]

        if citations_with_dates:
            # Sort by effective_date descending
            sorted_by_date = sorted(
                citations_with_dates,
                key=lambda c: self._parse_date(c.effective_date),
                reverse=True,
            )
            return sorted_by_date[0], "effective_date"

        # Try to resolve by version
        citations_with_versions = [
            c for c in conflicting_citations if c.version is not None
        ]

        if citations_with_versions:
            # Sort by version descending (assumes semantic versioning or simple comparison)
            sorted_by_version = sorted(
                citations_with_versions,
                key=lambda c: c.version or "",
                reverse=True,
            )
            return sorted_by_version[0], "version"

        # Fallback to highest score
        sorted_by_score = sorted(
            conflicting_citations,
            key=lambda c: c.score,
            reverse=True,
        )
        return sorted_by_score[0], "relevance_score"

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> datetime:
        """Parse date string to datetime for comparison.

        Args:
            date_str: Date string in various formats.

        Returns:
            Parsed datetime, or minimum datetime if parsing fails.
        """
        if not date_str:
            return datetime.min

        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return datetime.min


class FailSafePolicy:
    """Implements fail-safe policy for answer generation.

    Ensures that definitive answers are never provided when evidence
    is insufficient or conflicting.
    """

    # Messages for different scenarios
    INSUFFICIENT_EVIDENCE_MSG = (
        "현재 문서로는 확답을 드리기 어렵습니다. "
        "질문을 더 구체적으로 해주시거나, 다른 키워드로 검색해 보시기 바랍니다."
    )

    NO_RESULTS_MSG = (
        "관련 문서를 찾을 수 없습니다. "
        "다른 표현이나 키워드로 질문해 주세요."
    )

    CONFLICT_MSG = (
        "문서 간 상충되는 정보가 발견되었습니다. "
        "아래에 최신 정보를 기준으로 안내해 드리지만, 담당자에게 확인을 권장합니다."
    )

    @staticmethod
    def should_withhold_answer(grounding_result: GroundingResult) -> bool:
        """Determine if an answer should be withheld based on grounding result.

        Args:
            grounding_result: Result from grounding check.

        Returns:
            True if answer should be withheld.
        """
        # Always withhold for insufficient evidence or no results
        if grounding_result.verdict in (Verdict.NOT_FOUND, Verdict.INSUFFICIENT):
            return True

        # For conflicts, we provide a cautious answer with the recommended citation
        # but mark it as needing verification
        return False

    @staticmethod
    def get_withheld_message(grounding_result: GroundingResult) -> str:
        """Get appropriate message when answer is withheld.

        Args:
            grounding_result: Result from grounding check.

        Returns:
            User-friendly message explaining why answer is withheld.
        """
        if grounding_result.withheld_reason:
            return grounding_result.withheld_reason

        if grounding_result.verdict == Verdict.NOT_FOUND:
            return FailSafePolicy.NO_RESULTS_MSG

        if grounding_result.verdict == Verdict.INSUFFICIENT:
            return FailSafePolicy.INSUFFICIENT_EVIDENCE_MSG

        if grounding_result.verdict == Verdict.CONFLICT:
            return FailSafePolicy.CONFLICT_MSG

        return "답변을 생성할 수 없습니다."
