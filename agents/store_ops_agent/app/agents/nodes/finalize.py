"""Finalize Node for StoreOps Agent.

This node:
1. Takes grounding check result and draft answer
2. Constructs final response based on verdict
3. Generates follow-up questions if evidence insufficient
4. Returns final response with answer, citations, follow_up_question, meta
"""

import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.state import (
    AgentState,
    FinalResponse,
    GroundingVerdict,
)
from app.models.schemas import Citation

logger = logging.getLogger(__name__)


class FollowUpQuestions(BaseModel):
    """Schema for follow-up questions generation."""

    questions: list[str] = Field(
        min_length=1,
        max_length=3,
        description="1-3 follow-up questions to clarify the user's intent",
    )
    clarification_needed: str = Field(
        description="Brief explanation of why clarification is needed"
    )


FOLLOWUP_SYSTEM_PROMPT = """You are a helpful assistant that generates follow-up questions when the original question cannot be fully answered.

Your task:
1. Generate 1-3 clarifying questions that would help provide a better answer
2. Questions should be specific and actionable
3. Write in Korean (한국어)

Examples of good follow-up questions:
- "어느 매장 유형(카페, 편의점 등)에 대해 알고 싶으신가요?"
- "해당 정책이 적용되는 시점을 알려주시겠어요?"
- "환불 사유가 어떻게 되나요?"

Keep questions concise and relevant to the original question."""


FOLLOWUP_USER_TEMPLATE = """원래 질문: {question}

근거 부족 이유: {issues}

위 질문에 대해 더 정확한 답변을 드리기 위한 추가 질문을 생성해주세요."""


async def finalize_node(state: AgentState) -> dict[str, Any]:
    """Finalize the response based on grounding check.

    Args:
        state: Current agent state with grounding result.

    Returns:
        Dictionary with final response.
    """
    start_time = time.time()

    grounding = state.get("grounding")
    draft = state.get("draft")
    retrieval = state.get("retrieval")

    # Handle missing data
    if grounding is None or draft is None:
        return _create_error_response(state, "Missing grounding or draft data")

    # Get citations used in the answer
    citations = _get_final_citations(
        draft.citations_used,
        retrieval.candidates if retrieval else [],
    )

    # Build response based on verdict
    if grounding.verdict == GroundingVerdict.PASS:
        final = await _create_success_response(draft, citations)
    elif grounding.verdict == GroundingVerdict.INSUFFICIENT:
        final = await _create_insufficient_response(state, draft, citations, grounding)
    elif grounding.verdict == GroundingVerdict.CONFLICT:
        final = await _create_conflict_response(state, draft, citations, grounding)
    elif grounding.verdict == GroundingVerdict.OFF_TOPIC:
        final = _create_off_topic_response()
    else:
        final = await _create_insufficient_response(state, draft, citations, grounding)

    latency_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Finalize completed: withheld={final.withheld}, "
        f"citations={len(final.citations)}, latency={latency_ms:.1f}ms"
    )

    # Calculate total latency
    current_metrics = state.get("metrics")
    if current_metrics:
        current_metrics.total_latency_ms = (
            current_metrics.parse_latency_ms
            + current_metrics.retrieve_latency_ms
            + current_metrics.generate_latency_ms
            + current_metrics.grounding_latency_ms
            + latency_ms
        )

    return {
        "final": final,
        "metrics": current_metrics,
    }


def _get_final_citations(
    citation_ids: list[str], candidates: list[Citation]
) -> list[Citation]:
    """Get final Citation objects for response.

    Args:
        citation_ids: List of chunk_ids used.
        candidates: All candidates.

    Returns:
        List of Citation objects.
    """
    if not candidates:
        return []

    id_to_citation = {c.chunk_id: c for c in candidates}
    return [id_to_citation[cid] for cid in citation_ids if cid in id_to_citation]


async def _create_success_response(
    draft: Any, citations: list[Citation]
) -> FinalResponse:
    """Create successful response with grounded answer.

    Args:
        draft: Draft answer.
        citations: Supporting citations.

    Returns:
        Final response.
    """
    return FinalResponse(
        answer=draft.content,
        citations=citations,
        follow_up_question=None,
        withheld=False,
        withheld_reason=None,
    )


async def _create_insufficient_response(
    state: AgentState,
    draft: Any,
    citations: list[Citation],
    grounding: Any,
) -> FinalResponse:
    """Create response when evidence is insufficient.

    Args:
        state: Current state.
        draft: Draft answer.
        citations: Available citations.
        grounding: Grounding result.

    Returns:
        Final response with follow-up question.
    """
    question = state["question"]
    issues = grounding.issues

    # Generate follow-up questions
    follow_up = await _generate_followup_questions(question, issues)

    # Decide whether to withhold or provide partial answer
    counters = state.get("counters")
    max_attempts_reached = counters and counters.retrieval_attempts >= 2

    if max_attempts_reached or grounding.evidence_coverage < 0.3:
        # Withhold answer completely
        return FinalResponse(
            answer="현재 문서로는 확답을 드리기 어렵습니다.",
            citations=citations,
            follow_up_question=follow_up,
            withheld=True,
            withheld_reason="관련 문서의 근거가 충분하지 않습니다.",
        )
    else:
        # Provide partial answer with disclaimer
        return FinalResponse(
            answer=f"참고: 아래 답변은 제한된 근거에 기반합니다.\n\n{draft.content}",
            citations=citations,
            follow_up_question=follow_up,
            withheld=False,
            withheld_reason=None,
        )


async def _create_conflict_response(
    state: AgentState,
    draft: Any,
    citations: list[Citation],
    grounding: Any,
) -> FinalResponse:
    """Create response when documents conflict.

    Args:
        state: Current state.
        draft: Draft answer.
        citations: Available citations.
        grounding: Grounding result.

    Returns:
        Final response noting conflict.
    """
    # Find most recent citation for recommendation
    recommended = _find_most_recent_citation(citations)

    conflict_notice = (
        "주의: 문서 간 상충되는 정보가 발견되었습니다. "
        "아래는 가장 최신 문서를 기준으로 한 답변입니다. "
        "정확한 정보는 담당자에게 확인해 주세요.\n\n"
    )

    return FinalResponse(
        answer=conflict_notice + draft.content,
        citations=[recommended] if recommended else citations,
        follow_up_question=None,
        withheld=False,
        withheld_reason="문서 간 상충",
    )


def _create_off_topic_response() -> FinalResponse:
    """Create response for off-topic queries.

    Returns:
        Final response for out-of-scope.
    """
    return FinalResponse(
        answer="죄송합니다. 해당 질문은 매장 운영과 관련되지 않아 답변드리기 어렵습니다. "
        "매장 운영, 정책, 절차에 관한 질문을 해주세요.",
        citations=[],
        follow_up_question=None,
        withheld=True,
        withheld_reason="범위 외 질문",
    )


def _create_error_response(state: AgentState, error: str) -> dict[str, Any]:
    """Create error response.

    Args:
        state: Current state.
        error: Error message.

    Returns:
        Response dictionary with error.
    """
    errors = state.get("errors", [])
    errors.append(f"Finalize error: {error}")

    return {
        "final": FinalResponse(
            answer="죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
            citations=[],
            follow_up_question=None,
            withheld=True,
            withheld_reason=error,
        ),
        "errors": errors,
    }


async def _generate_followup_questions(
    question: str, issues: list[str]
) -> str | None:
    """Generate follow-up questions using LLM.

    Args:
        question: Original question.
        issues: List of grounding issues.

    Returns:
        Follow-up question or None.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        structured_llm = llm.with_structured_output(FollowUpQuestions)

        user_message = FOLLOWUP_USER_TEMPLATE.format(
            question=question,
            issues="\n".join(issues) if issues else "근거 부족",
        )

        result: FollowUpQuestions = await structured_llm.ainvoke(
            [
                {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        # Return first question as the follow-up
        if result.questions:
            return result.questions[0]
        return None

    except Exception as e:
        logger.warning(f"Failed to generate follow-up questions: {e}")
        # Fallback generic question
        return "질문을 더 구체적으로 해주시겠어요?"


def _find_most_recent_citation(citations: list[Citation]) -> Citation | None:
    """Find citation with most recent effective_date.

    Args:
        citations: List of citations.

    Returns:
        Most recent citation or None.
    """
    if not citations:
        return None

    citations_with_dates = [c for c in citations if c.effective_date]
    if not citations_with_dates:
        # Return highest score if no dates
        return max(citations, key=lambda c: c.score)

    return max(citations_with_dates, key=lambda c: c.effective_date or "")
