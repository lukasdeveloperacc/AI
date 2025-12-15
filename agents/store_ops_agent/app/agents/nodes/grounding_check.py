"""Grounding Check Node for StoreOps Agent.

This node:
1. Takes draft answer and citations from state
2. Verifies that the answer is grounded in the citations
3. Checks for evidence coverage and conflicts
4. Returns verdict: pass/insufficient/conflict/off_topic
"""

import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.state import (
    AgentState,
    GroundingCheckResult,
    GroundingVerdict,
)
from app.models.schemas import Citation

logger = logging.getLogger(__name__)

# Thresholds
MIN_CITATIONS_REQUIRED = 2
MIN_EVIDENCE_COVERAGE = 0.6


class GroundingEvaluation(BaseModel):
    """Schema for LLM grounding evaluation response."""

    is_grounded: bool = Field(
        description="Whether the answer is fully supported by the citations"
    )
    evidence_coverage: float = Field(
        ge=0.0, le=1.0, description="Percentage of answer claims supported by evidence"
    )
    has_conflicts: bool = Field(
        description="Whether there are conflicting statements in the citations"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List of claims in the answer not supported by citations",
    )
    reasoning: str = Field(description="Brief explanation of the evaluation")


GROUNDING_SYSTEM_PROMPT = """You are a fact-checking assistant that evaluates whether an answer is properly grounded in source documents.

Your task:
1. Check if EVERY claim in the answer is supported by the provided citations
2. Calculate evidence coverage (0.0-1.0): what percentage of claims are supported
3. Detect any conflicts between citations
4. List any unsupported claims

Evaluation criteria:
- is_grounded: True only if ALL major claims are supported by citations
- evidence_coverage: 1.0 if fully supported, lower for partial support
- has_conflicts: True if citations contain contradictory information

Be strict in your evaluation. If information is implied but not explicitly stated, consider it unsupported."""


GROUNDING_USER_TEMPLATE = """답변:
{answer}

사용된 인용문:
{citations}

위 답변이 인용문에 의해 적절히 뒷받침되는지 평가해주세요."""


async def grounding_check_node(state: AgentState) -> dict[str, Any]:
    """Check if draft answer is properly grounded in citations.

    Args:
        state: Current agent state with draft answer.

    Returns:
        Dictionary with grounding check result.
    """
    start_time = time.time()

    draft = state.get("draft")
    retrieval = state.get("retrieval")

    # Handle edge cases
    if draft is None:
        return _create_insufficient_result(
            state, start_time, "No draft answer to evaluate"
        )

    if draft.model in ["none", "out_of_scope", "fallback"]:
        return _handle_special_draft(state, draft, start_time)

    if retrieval is None or not retrieval.candidates:
        return _create_insufficient_result(
            state, start_time, "No citations available for grounding"
        )

    # Check minimum citations
    used_citations = _get_used_citations(draft.citations_used, retrieval.candidates)
    if len(used_citations) < MIN_CITATIONS_REQUIRED:
        return _create_insufficient_result(
            state,
            start_time,
            f"Insufficient citations: {len(used_citations)} < {MIN_CITATIONS_REQUIRED}",
        )

    try:
        # Build citations context
        citations_text = _build_citations_text(used_citations)

        # Evaluate grounding using LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(GroundingEvaluation)

        user_message = GROUNDING_USER_TEMPLATE.format(
            answer=draft.content,
            citations=citations_text,
        )

        result: GroundingEvaluation = await structured_llm.ainvoke(
            [
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        # Determine verdict
        verdict, issues, action = _determine_verdict(result, len(used_citations))

        grounding_result = GroundingCheckResult(
            verdict=verdict,
            evidence_coverage=result.evidence_coverage,
            issues=issues,
            recommended_action=action,
        )

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Grounding check completed: verdict={verdict.value}, "
            f"coverage={result.evidence_coverage:.2f}, latency={latency_ms:.1f}ms"
        )

        # Update metrics
        current_metrics = state.get("metrics")
        if current_metrics:
            current_metrics.grounding_latency_ms = latency_ms

        return {
            "grounding": grounding_result,
            "metrics": current_metrics,
        }

    except Exception as e:
        logger.error(f"Grounding check failed: {e}")
        errors = state.get("errors", [])
        errors.append(f"Grounding check error: {e}")

        # Fallback: assume insufficient if we can't verify
        return {
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.INSUFFICIENT,
                evidence_coverage=0.0,
                issues=[f"Evaluation failed: {e}"],
                recommended_action="retry",
            ),
            "errors": errors,
        }


def _get_used_citations(
    citation_ids: list[str], candidates: list[Citation]
) -> list[Citation]:
    """Get Citation objects for used citation IDs.

    Args:
        citation_ids: List of chunk_ids used in the answer.
        candidates: All available candidates.

    Returns:
        List of Citation objects that were used.
    """
    id_to_citation = {c.chunk_id: c for c in candidates}
    return [id_to_citation[cid] for cid in citation_ids if cid in id_to_citation]


def _build_citations_text(citations: list[Citation]) -> str:
    """Build citations text for evaluation.

    Args:
        citations: List of citations used.

    Returns:
        Formatted citations text.
    """
    parts = []
    for i, citation in enumerate(citations, 1):
        parts.append(
            f"[인용 {i}] (chunk_id: {citation.chunk_id})\n"
            f"출처: {citation.title}\n"
            f"내용: {citation.snippet}\n"
        )
    return "\n".join(parts)


def _determine_verdict(
    evaluation: GroundingEvaluation, num_citations: int
) -> tuple[GroundingVerdict, list[str], str | None]:
    """Determine grounding verdict from evaluation.

    Args:
        evaluation: LLM evaluation result.
        num_citations: Number of citations used.

    Returns:
        Tuple of (verdict, issues list, recommended action).
    """
    issues = []

    # Check for conflicts first
    if evaluation.has_conflicts:
        issues.append("문서 간 상충되는 정보가 발견되었습니다")
        return GroundingVerdict.CONFLICT, issues, "resolve_conflict"

    # Check evidence coverage
    if evaluation.evidence_coverage < MIN_EVIDENCE_COVERAGE:
        issues.append(f"근거 범위 부족: {evaluation.evidence_coverage:.0%}")
        issues.extend(evaluation.unsupported_claims)
        return GroundingVerdict.INSUFFICIENT, issues, "retry_retrieval"

    # Check if grounded
    if not evaluation.is_grounded:
        issues.append("일부 주장이 근거로 뒷받침되지 않습니다")
        issues.extend(evaluation.unsupported_claims)
        return GroundingVerdict.INSUFFICIENT, issues, "refine_answer"

    # Check citation count
    if num_citations < MIN_CITATIONS_REQUIRED:
        issues.append(f"인용 수 부족: {num_citations}")
        return GroundingVerdict.INSUFFICIENT, issues, "retry_retrieval"

    # All checks passed
    return GroundingVerdict.PASS, [], None


def _handle_special_draft(
    state: AgentState, draft: Any, start_time: float
) -> dict[str, Any]:
    """Handle special draft types (none, out_of_scope, fallback).

    Args:
        state: Current agent state.
        draft: Draft answer.
        start_time: Start time.

    Returns:
        Appropriate grounding result.
    """
    latency_ms = (time.time() - start_time) * 1000

    current_metrics = state.get("metrics")
    if current_metrics:
        current_metrics.grounding_latency_ms = latency_ms

    if draft.model == "out_of_scope":
        return {
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.OFF_TOPIC,
                evidence_coverage=0.0,
                issues=["Query is out of scope"],
                recommended_action=None,
            ),
            "metrics": current_metrics,
        }

    if draft.model in ["none", "fallback"]:
        return {
            "grounding": GroundingCheckResult(
                verdict=GroundingVerdict.INSUFFICIENT,
                evidence_coverage=0.0,
                issues=["No proper context available"],
                recommended_action="retry_retrieval",
            ),
            "metrics": current_metrics,
        }

    return {
        "grounding": GroundingCheckResult(
            verdict=GroundingVerdict.INSUFFICIENT,
            evidence_coverage=0.0,
            issues=["Unknown draft type"],
        ),
        "metrics": current_metrics,
    }


def _create_insufficient_result(
    state: AgentState, start_time: float, reason: str
) -> dict[str, Any]:
    """Create insufficient grounding result.

    Args:
        state: Current agent state.
        start_time: Start time.
        reason: Reason for insufficiency.

    Returns:
        Grounding result dictionary.
    """
    latency_ms = (time.time() - start_time) * 1000

    current_metrics = state.get("metrics")
    if current_metrics:
        current_metrics.grounding_latency_ms = latency_ms

    return {
        "grounding": GroundingCheckResult(
            verdict=GroundingVerdict.INSUFFICIENT,
            evidence_coverage=0.0,
            issues=[reason],
            recommended_action="retry_retrieval",
        ),
        "metrics": current_metrics,
    }
