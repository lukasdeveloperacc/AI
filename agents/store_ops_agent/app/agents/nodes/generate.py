"""Generate Node for StoreOps Agent.

This node:
1. Takes retrieved candidates from state
2. Generates a draft answer using LLM (3-5 sentences)
3. Produces 2-3 citations referencing source documents
4. Returns draft answer with citations used
"""

import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.state import (
    AgentState,
    DraftAnswer,
    Intent,
)
from app.models.schemas import Citation

logger = logging.getLogger(__name__)


class GeneratedAnswer(BaseModel):
    """Schema for LLM generated answer response."""

    answer: str = Field(
        description="Generated answer in 3-5 sentences based on retrieved documents"
    )
    citations_used: list[str] = Field(
        description="List of chunk_ids used to generate the answer (2-3 citations)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the generated answer"
    )


GENERATION_SYSTEM_PROMPT = """You are a helpful store operations assistant that answers questions based on provided documents.

Instructions:
1. Answer the question in 3-5 sentences using ONLY information from the provided context.
2. Use 2-3 citations from the context to support your answer.
3. If the context doesn't contain enough information, indicate uncertainty.
4. Write in Korean (한국어) unless the user asks in another language.
5. Be concise and focus on the most relevant information.

Context documents will be provided with their chunk_ids. Reference these in your citations_used.

Important:
- Do NOT make up information not present in the context
- If documents conflict, mention the discrepancy
- Provide practical, actionable information when possible"""


GENERATION_USER_TEMPLATE = """질문: {question}

검색된 문서:
{context}

위 문서를 기반으로 질문에 답변해주세요. 답변에 사용한 문서의 chunk_id를 citations_used에 포함해주세요."""


async def generate_node(state: AgentState) -> dict[str, Any]:
    """Generate draft answer from retrieved documents.

    Args:
        state: Current agent state with retrieval results.

    Returns:
        Dictionary with draft answer.
    """
    start_time = time.time()

    # Get retrieval results
    retrieval = state.get("retrieval")
    if retrieval is None or not retrieval.candidates:
        return _create_no_context_response(state, start_time)

    question = state["question"]
    parsed = state.get("parsed")

    # Check if query is out of scope
    if parsed and parsed.intent == Intent.OUT_OF_SCOPE:
        return _create_out_of_scope_response(state, start_time)

    try:
        # Build context from candidates
        context = _build_context(retrieval.candidates)

        # Generate answer using LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        structured_llm = llm.with_structured_output(GeneratedAnswer)

        user_message = GENERATION_USER_TEMPLATE.format(
            question=question,
            context=context,
        )

        result: GeneratedAnswer = await structured_llm.ainvoke(
            [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        # Validate citations exist in candidates
        valid_citations = _validate_citations(result.citations_used, retrieval.candidates)

        draft = DraftAnswer(
            content=result.answer,
            citations_used=valid_citations,
            model="gpt-4o-mini",
        )

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Generate completed: {len(valid_citations)} citations, "
            f"latency={latency_ms:.1f}ms"
        )

        # Update metrics
        current_metrics = state.get("metrics")
        if current_metrics:
            current_metrics.generate_latency_ms = latency_ms

        return {
            "draft": draft,
            "metrics": current_metrics,
        }

    except Exception as e:
        logger.error(f"Generate failed: {e}")
        errors = state.get("errors", [])
        errors.append(f"Generate error: {e}")

        # Fallback to simple extraction
        draft = _create_fallback_draft(retrieval.candidates)

        return {
            "draft": draft,
            "errors": errors,
        }


def _build_context(candidates: list[Citation]) -> str:
    """Build context string from candidates.

    Args:
        candidates: List of citation candidates.

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, citation in enumerate(candidates, 1):
        context_parts.append(
            f"[문서 {i}]\n"
            f"- chunk_id: {citation.chunk_id}\n"
            f"- 제목: {citation.title}\n"
            f"- 내용: {citation.snippet}\n"
            f"- 관련도 점수: {citation.score:.2f}\n"
        )

    return "\n".join(context_parts)


def _validate_citations(
    citations_used: list[str], candidates: list[Citation]
) -> list[str]:
    """Validate that cited chunk_ids exist in candidates.

    Args:
        citations_used: List of chunk_ids claimed to be used.
        candidates: List of actual candidates.

    Returns:
        List of valid chunk_ids.
    """
    valid_chunk_ids = {c.chunk_id for c in candidates}
    validated = [cid for cid in citations_used if cid in valid_chunk_ids]

    # If no valid citations, use top candidates
    if not validated and candidates:
        validated = [c.chunk_id for c in candidates[:2]]

    return validated


def _create_fallback_draft(candidates: list[Citation]) -> DraftAnswer:
    """Create fallback draft from candidates without LLM.

    Args:
        candidates: List of citation candidates.

    Returns:
        Simple draft answer.
    """
    if not candidates:
        return DraftAnswer(
            content="관련 문서를 찾을 수 없습니다.",
            citations_used=[],
            model="fallback",
        )

    # Use top candidate's snippet as basis
    top_candidate = candidates[0]
    answer = f"검색된 문서에 따르면: {top_candidate.snippet}"

    return DraftAnswer(
        content=answer,
        citations_used=[candidates[0].chunk_id] if candidates else [],
        model="fallback",
    )


def _create_no_context_response(
    state: AgentState, start_time: float
) -> dict[str, Any]:
    """Create response when no context is available.

    Args:
        state: Current agent state.
        start_time: Start time for latency calculation.

    Returns:
        Dictionary with empty draft.
    """
    latency_ms = (time.time() - start_time) * 1000

    current_metrics = state.get("metrics")
    if current_metrics:
        current_metrics.generate_latency_ms = latency_ms

    draft = DraftAnswer(
        content="관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해 주세요.",
        citations_used=[],
        model="none",
    )

    return {
        "draft": draft,
        "metrics": current_metrics,
    }


def _create_out_of_scope_response(
    state: AgentState, start_time: float
) -> dict[str, Any]:
    """Create response for out-of-scope queries.

    Args:
        state: Current agent state.
        start_time: Start time for latency calculation.

    Returns:
        Dictionary with out-of-scope draft.
    """
    latency_ms = (time.time() - start_time) * 1000

    current_metrics = state.get("metrics")
    if current_metrics:
        current_metrics.generate_latency_ms = latency_ms

    draft = DraftAnswer(
        content="죄송합니다. 해당 질문은 매장 운영과 관련되지 않아 답변드리기 어렵습니다. "
        "매장 운영, 정책, 절차에 관한 질문을 해주세요.",
        citations_used=[],
        model="out_of_scope",
    )

    return {
        "draft": draft,
        "metrics": current_metrics,
    }
