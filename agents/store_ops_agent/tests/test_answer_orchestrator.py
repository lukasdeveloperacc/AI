"""Tests for answer orchestrator with retry logic."""

import pytest

from app.core.answer_orchestrator import AnswerOrchestrator, OrchestratorConfig
from app.core.tracing import RequestTracer
from app.models.schemas import Citation, Verdict


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, citations_sequence: list[list[Citation]]) -> None:
        """Initialize with a sequence of citation lists to return.

        Args:
            citations_sequence: List of citation lists, one per retrieval call.
        """
        self.citations_sequence = citations_sequence
        self.call_count = 0

    async def retrieve(self, query: str, topk: int) -> list[Citation]:
        """Return the next set of citations in the sequence."""
        if self.call_count < len(self.citations_sequence):
            result = self.citations_sequence[self.call_count]
            self.call_count += 1
            return result
        return []


@pytest.fixture
def config() -> OrchestratorConfig:
    """Create orchestrator config."""
    return OrchestratorConfig(
        max_retrieval_attempts=2,
        min_citations=1,
        min_score_threshold=0.6,
        max_followup_questions=3,
    )


@pytest.fixture
def orchestrator(config: OrchestratorConfig) -> AnswerOrchestrator:
    """Create answer orchestrator."""
    return AnswerOrchestrator(config=config)


@pytest.fixture
def high_score_citation() -> Citation:
    """Create a high-score citation."""
    return Citation(
        doc_id="doc_001",
        title="매장 운영 매뉴얼",
        chunk_id="chunk_001",
        snippet="영업시간은 오전 9시부터 오후 10시까지입니다.",
        score=0.92,
    )


@pytest.fixture
def low_score_citation() -> Citation:
    """Create a low-score citation."""
    return Citation(
        doc_id="doc_002",
        title="직원 근무 규정",
        chunk_id="chunk_002",
        snippet="근무 관련 정보입니다.",
        score=0.45,
    )


class TestAnswerOrchestrator:
    """Tests for AnswerOrchestrator class."""

    @pytest.mark.asyncio
    async def test_successful_retrieval_no_retry(
        self,
        orchestrator: AnswerOrchestrator,
        high_score_citation: Citation,
    ) -> None:
        """Test successful retrieval without retry."""
        tracer = RequestTracer()
        retriever = MockRetriever([[high_score_citation]])

        response = await orchestrator.process_question(
            question="영업시간이 어떻게 되나요?",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        assert response.meta.verdict == Verdict.ANSWERED
        assert response.withheld is False
        assert retriever.call_count == 1
        assert tracer.retrieval_attempts == 1

    @pytest.mark.asyncio
    async def test_retry_on_no_results(
        self,
        orchestrator: AnswerOrchestrator,
        high_score_citation: Citation,
    ) -> None:
        """Test retry when first retrieval returns no results."""
        tracer = RequestTracer()
        # First call returns empty, second returns results
        retriever = MockRetriever([[], [high_score_citation]])

        response = await orchestrator.process_question(
            question="영업시간이 어떻게 되나요?",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        assert response.meta.verdict == Verdict.ANSWERED
        assert response.withheld is False
        assert retriever.call_count == 2
        assert tracer.retrieval_attempts == 2

    @pytest.mark.asyncio
    async def test_retry_on_low_score_results(
        self,
        orchestrator: AnswerOrchestrator,
        low_score_citation: Citation,
        high_score_citation: Citation,
    ) -> None:
        """Test retry when first retrieval returns low-score results."""
        tracer = RequestTracer()
        # First call returns low score, second returns high score
        retriever = MockRetriever([[low_score_citation], [high_score_citation]])

        response = await orchestrator.process_question(
            question="영업시간이 어떻게 되나요?",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        assert response.meta.verdict == Verdict.ANSWERED
        assert retriever.call_count == 2

    @pytest.mark.asyncio
    async def test_max_one_retry(
        self,
        orchestrator: AnswerOrchestrator,
    ) -> None:
        """Test that retry is limited to max 1 (total 2 attempts)."""
        tracer = RequestTracer()
        # All calls return empty - should only try twice
        retriever = MockRetriever([[], [], []])

        response = await orchestrator.process_question(
            question="영업시간이 어떻게 되나요?",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        assert response.meta.verdict == Verdict.NOT_FOUND
        assert response.withheld is True
        assert retriever.call_count == 2  # Max 2 attempts
        assert tracer.retrieval_attempts == 2

    @pytest.mark.asyncio
    async def test_withheld_answer_for_insufficient(
        self,
        orchestrator: AnswerOrchestrator,
        low_score_citation: Citation,
    ) -> None:
        """Test answer is withheld for insufficient evidence."""
        tracer = RequestTracer()
        retriever = MockRetriever([[low_score_citation], [low_score_citation]])

        response = await orchestrator.process_question(
            question="영업시간이 어떻게 되나요?",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        assert response.meta.verdict == Verdict.INSUFFICIENT
        assert response.withheld is True
        assert response.withheld_reason is not None

    @pytest.mark.asyncio
    async def test_follow_up_questions_generated(
        self,
        orchestrator: AnswerOrchestrator,
    ) -> None:
        """Test that follow-up questions are generated."""
        tracer = RequestTracer()
        retriever = MockRetriever([[]])

        response = await orchestrator.process_question(
            question="영업시간이 어떻게 되나요?",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        assert response.follow_up_questions is not None
        assert len(response.follow_up_questions) > 0

    @pytest.mark.asyncio
    async def test_conflict_handling(
        self,
        orchestrator: AnswerOrchestrator,
    ) -> None:
        """Test handling of conflicting citations."""
        tracer = RequestTracer()

        conflicting_citations = [
            Citation(
                doc_id="doc_001",
                title="매뉴얼 2024",
                chunk_id="chunk_001",
                snippet="영업시간: 9시-22시",
                score=0.90,
                effective_date="2024-01-01",
            ),
            Citation(
                doc_id="doc_002",
                title="매뉴얼 2023",
                chunk_id="chunk_002",
                snippet="영업시간: 10시-21시",
                score=0.88,
                effective_date="2023-01-01",
            ),
        ]
        retriever = MockRetriever([conflicting_citations])

        response = await orchestrator.process_question(
            question="영업시간이 어떻게 되나요?",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        assert response.meta.verdict == Verdict.CONFLICT
        assert response.meta.conflict_info is not None
        assert response.withheld is False  # Conflict provides cautious answer
        assert "상충" in response.answer

    @pytest.mark.asyncio
    async def test_no_retry_for_conflict(
        self,
        orchestrator: AnswerOrchestrator,
    ) -> None:
        """Test that retry is not performed for conflict verdict."""
        tracer = RequestTracer()

        conflicting_citations = [
            Citation(
                doc_id="doc_001",
                title="문서 A",
                chunk_id="chunk_001",
                snippet="정보 A",
                score=0.85,
                effective_date="2024-01-01",
            ),
            Citation(
                doc_id="doc_002",
                title="문서 B",
                chunk_id="chunk_002",
                snippet="정보 B",
                score=0.84,
                effective_date="2023-01-01",
            ),
        ]
        # Second set would be returned if retry happened
        other_citations = [
            Citation(
                doc_id="doc_003",
                title="다른 문서",
                chunk_id="chunk_003",
                snippet="다른 정보",
                score=0.95,
            ),
        ]
        retriever = MockRetriever([conflicting_citations, other_citations])

        response = await orchestrator.process_question(
            question="질문",
            topk=5,
            tracer=tracer,
            retriever=retriever,
        )

        # Should not retry for conflict - only 1 call
        assert retriever.call_count == 1
        assert response.meta.verdict == Verdict.CONFLICT


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = OrchestratorConfig()

        assert config.max_retrieval_attempts == 2
        assert config.min_citations == 1
        assert config.min_score_threshold == 0.6
        assert config.max_followup_questions == 3

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = OrchestratorConfig(
            max_retrieval_attempts=3,
            min_citations=2,
            min_score_threshold=0.7,
            max_followup_questions=5,
        )

        assert config.max_retrieval_attempts == 3
        assert config.min_citations == 2
        assert config.min_score_threshold == 0.7
        assert config.max_followup_questions == 5


class TestAnswerGeneration:
    """Tests for answer generation methods."""

    def test_default_answer_generation(
        self,
        orchestrator: AnswerOrchestrator,
        high_score_citation: Citation,
    ) -> None:
        """Test default answer generation."""
        answer = orchestrator._generate_default_answer(
            question="영업시간이 어떻게 되나요?",
            citations=[high_score_citation],
        )

        assert "영업시간" in answer
        assert high_score_citation.title in answer

    def test_default_answer_no_citations(
        self,
        orchestrator: AnswerOrchestrator,
    ) -> None:
        """Test default answer with no citations."""
        answer = orchestrator._generate_default_answer(
            question="영업시간이 어떻게 되나요?",
            citations=[],
        )

        assert "찾을 수 없습니다" in answer

    def test_conflict_answer_generation(
        self,
        orchestrator: AnswerOrchestrator,
    ) -> None:
        """Test conflict answer generation."""
        from app.models.schemas import ConflictInfo

        citation = Citation(
            doc_id="doc_001",
            title="최신 매뉴얼",
            chunk_id="chunk_001",
            snippet="최신 정보입니다.",
            score=0.90,
            effective_date="2024-01-01",
        )
        conflict_info = ConflictInfo(
            conflicting_citations=["doc_001", "doc_002"],
            resolution_basis="effective_date",
            recommended_citation_id="doc_001",
        )

        answer = orchestrator._generate_conflict_answer(
            question="질문",
            recommended_citation=citation,
            conflict_info=conflict_info,
        )

        assert "상충" in answer
        assert citation.title in answer
        assert "2024-01-01" in answer
