"""Answer orchestrator with retry logic and fail-safe handling.

This module coordinates the answer generation process, including:
- Document retrieval with retry logic
- Grounding check for evidence validation
- Follow-up question generation
- Fail-safe policy enforcement
"""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from app.core.followup_generator import FollowUpGenerator
from app.core.grounding_checker import FailSafePolicy, GroundingChecker, GroundingResult
from app.core.tracing import RequestTracer
from app.models.schemas import AnswerResponse, Citation, ConflictInfo, Meta, Verdict


class RetrieverProtocol(Protocol):
    """Protocol for document retrieval."""

    async def retrieve(
        self,
        query: str,
        topk: int,
    ) -> list[Citation]:
        """Retrieve relevant documents for a query."""
        ...


@dataclass
class OrchestratorConfig:
    """Configuration for the answer orchestrator."""

    max_retrieval_attempts: int = 2  # Max 1 retry = 2 total attempts
    min_citations: int = 1
    min_score_threshold: float = 0.6
    max_followup_questions: int = 3


class AnswerOrchestrator:
    """Orchestrates the answer generation process with fail-safe handling.

    This class coordinates:
    1. Document retrieval (with max 1 retry)
    2. Grounding check for evidence validation
    3. Follow-up question generation when evidence is insufficient
    4. Answer generation or withholding based on fail-safe policy
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Configuration for the orchestrator.
        """
        self.config = config or OrchestratorConfig()
        self.grounding_checker = GroundingChecker(
            min_citations=self.config.min_citations,
            min_score_threshold=self.config.min_score_threshold,
        )
        self.followup_generator = FollowUpGenerator(
            max_questions=self.config.max_followup_questions,
        )

    async def process_question(
        self,
        question: str,
        topk: int,
        tracer: RequestTracer,
        retriever: RetrieverProtocol,
        answer_generator: Optional[Callable[[str, list[Citation]], str]] = None,
    ) -> AnswerResponse:
        """Process a question with retry logic and fail-safe handling.

        Args:
            question: The user's question.
            topk: Number of documents to retrieve.
            tracer: Request tracer for metrics.
            retriever: Document retriever implementation.
            answer_generator: Optional function to generate answer from citations.

        Returns:
            AnswerResponse with answer, citations, and metadata.
        """
        citations: list[Citation] = []
        grounding_result: Optional[GroundingResult] = None
        attempt = 0

        # Retrieval loop with max 1 retry
        while attempt < self.config.max_retrieval_attempts:
            attempt += 1
            tracer.record_retrieval_attempt()

            # Retrieve documents
            citations = await retriever.retrieve(question, topk)

            # Perform grounding check
            grounding_result = self.grounding_checker.check(citations, question)

            # If we have sufficient evidence, break out of retry loop
            if grounding_result.is_sufficient:
                break

            # If it's the first attempt and evidence is insufficient (not conflict),
            # we can retry with modified query or expanded search
            if attempt == 1 and grounding_result.verdict in (
                Verdict.NOT_FOUND,
                Verdict.INSUFFICIENT,
            ):
                # Continue to retry
                continue

            # For conflicts or second attempt, don't retry further
            break

        # At this point, we have the final citations and grounding result
        if grounding_result is None:
            grounding_result = GroundingResult(
                verdict=Verdict.NOT_FOUND,
                is_sufficient=False,
                withheld_reason="검색을 수행할 수 없습니다.",
            )

        # Set verdict on tracer
        tracer.set_verdict(grounding_result.verdict)

        # Determine if answer should be withheld
        should_withhold = FailSafePolicy.should_withhold_answer(grounding_result)

        # Generate answer or withheld message
        if should_withhold:
            answer = FailSafePolicy.get_withheld_message(grounding_result)
        elif grounding_result.verdict == Verdict.CONFLICT and grounding_result.recommended_citation:
            # For conflicts, provide cautious answer based on recommended citation
            if answer_generator:
                answer = answer_generator(question, [grounding_result.recommended_citation])
            else:
                answer = self._generate_conflict_answer(
                    question,
                    grounding_result.recommended_citation,
                    grounding_result.conflict_info,
                )
        else:
            # Generate normal answer
            if answer_generator:
                answer = answer_generator(question, citations)
            else:
                answer = self._generate_default_answer(question, citations)

        # Generate follow-up questions
        follow_up_questions = self.followup_generator.generate(
            verdict=grounding_result.verdict,
            question=question,
            citations=citations,
        )

        # Build response
        return AnswerResponse(
            answer=answer,
            citations=citations,
            follow_up_questions=follow_up_questions if follow_up_questions else None,
            withheld=should_withhold,
            withheld_reason=grounding_result.withheld_reason if should_withhold else None,
            meta=self._build_meta(tracer, grounding_result),
        )

    def _generate_default_answer(
        self,
        question: str,
        citations: list[Citation],
    ) -> str:
        """Generate a default answer based on citations.

        This is a placeholder for actual LLM-based answer generation.

        Args:
            question: The user's question.
            citations: Retrieved citations.

        Returns:
            Generated answer string.
        """
        if not citations:
            return "관련 정보를 찾을 수 없습니다."

        # In real implementation, this would use an LLM
        top_citation = max(citations, key=lambda c: c.score)
        return (
            f"'{question}'에 대한 답변입니다. "
            f"'{top_citation.title}' 문서에 따르면, {top_citation.snippet} "
            "자세한 내용은 아래 근거 자료를 참고해 주세요."
        )

    def _generate_conflict_answer(
        self,
        question: str,
        recommended_citation: Citation,
        conflict_info: Optional[ConflictInfo],
    ) -> str:
        """Generate answer for conflict scenario.

        Args:
            question: The user's question.
            recommended_citation: The recommended citation after conflict resolution.
            conflict_info: Information about the conflict.

        Returns:
            Answer with conflict notice.
        """
        resolution_note = ""
        if conflict_info and conflict_info.resolution_basis:
            if conflict_info.resolution_basis == "effective_date":
                resolution_note = f"(적용일 {recommended_citation.effective_date} 기준)"
            elif conflict_info.resolution_basis == "version":
                resolution_note = f"(버전 {recommended_citation.version} 기준)"

        return (
            f"⚠️ 문서 간 상충되는 정보가 발견되었습니다. "
            f"최신 정보 {resolution_note}를 기준으로 안내해 드립니다.\n\n"
            f"'{recommended_citation.title}' 문서에 따르면: {recommended_citation.snippet}\n\n"
            "정확한 확인이 필요하시면 담당자에게 문의해 주세요."
        )

    def _build_meta(
        self,
        tracer: RequestTracer,
        grounding_result: GroundingResult,
    ) -> Meta:
        """Build Meta object from tracer and grounding result.

        Args:
            tracer: Request tracer with timing and attempt info.
            grounding_result: Result of grounding check.

        Returns:
            Meta object for response.
        """
        base_meta = tracer.to_meta()

        # Add conflict info if present
        if grounding_result.conflict_info:
            return Meta(
                trace_id=base_meta.trace_id,
                topk=base_meta.topk,
                retrieval_attempts=base_meta.retrieval_attempts,
                latency_ms=base_meta.latency_ms,
                verdict=grounding_result.verdict,
                conflict_info=grounding_result.conflict_info,
            )

        return Meta(
            trace_id=base_meta.trace_id,
            topk=base_meta.topk,
            retrieval_attempts=base_meta.retrieval_attempts,
            latency_ms=base_meta.latency_ms,
            verdict=grounding_result.verdict,
        )
