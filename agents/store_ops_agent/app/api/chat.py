"""Chat API endpoint for StoreOps Agent.

POST /chat endpoint that integrates with the LangGraph agent workflow.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Request

from app.agents.graph import run_agent
from app.agents.state import GroundingVerdict
from app.core.tracing import generate_trace_id
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    FilterInfo,
    Meta,
    Verdict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _map_grounding_verdict_to_verdict(grounding_verdict: GroundingVerdict) -> Verdict:
    """Map internal GroundingVerdict to API Verdict.

    Args:
        grounding_verdict: Internal verdict from grounding check.

    Returns:
        API-level Verdict enum value.
    """
    mapping = {
        GroundingVerdict.PASS: Verdict.ANSWERED,
        GroundingVerdict.INSUFFICIENT: Verdict.INSUFFICIENT,
        GroundingVerdict.CONFLICT: Verdict.CONFLICT,
        GroundingVerdict.OFF_TOPIC: Verdict.NOT_FOUND,
    }
    return mapping.get(grounding_verdict, Verdict.NOT_FOUND)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the StoreOps Agent",
    description="Send a question and receive an answer with citations from the knowledge base.",
    responses={
        200: {
            "description": "Successful response with answer and citations",
            "model": ChatResponse,
        },
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    },
)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Process a chat request through the LangGraph agent.

    This endpoint:
    1. Receives a question with optional filters
    2. Runs the full agent workflow (parse -> retrieve -> generate -> grounding -> finalize)
    3. Returns the answer with citations and metadata

    Args:
        request: The chat request containing the question and optional filters.
        http_request: FastAPI Request object for accessing request context.

    Returns:
        ChatResponse with answer, citations, follow-up question, and metadata.
    """
    start_time = time.perf_counter()
    trace_id = generate_trace_id()

    logger.info(
        f"[{trace_id}] Starting chat request",
        extra={
            "trace_id": trace_id,
            "question_length": len(request.question),
            "has_filters": any([
                request.store_type,
                request.category,
                request.effective_date,
                request.language,
            ]),
        },
    )

    # Run the agent workflow
    final_state = await run_agent(
        question=request.question,
        trace_id=trace_id,
        topk=8,  # Default topk for chat endpoint
        store_type=request.store_type,
        category=request.category,
        effective_date=request.effective_date,
        language=request.language,
    )

    # Extract results from final state
    final_response = final_state.get("final")
    counters = final_state.get("counters")
    metrics = final_state.get("metrics")
    grounding = final_state.get("grounding")
    filters = final_state.get("filters")

    # Calculate total latency
    total_latency_ms = (time.perf_counter() - start_time) * 1000

    # Build citations from final response
    citations: list[Citation] = []
    if final_response and final_response.citations:
        citations = final_response.citations

    # Determine verdict
    verdict = Verdict.NOT_FOUND
    if grounding:
        verdict = _map_grounding_verdict_to_verdict(grounding.verdict)
    elif final_response and not final_response.withheld:
        verdict = Verdict.ANSWERED if citations else Verdict.NOT_FOUND

    # Build filter info
    filter_info: Optional[FilterInfo] = None
    if filters and any([
        filters.category,
        filters.store_type,
        filters.effective_date,
        filters.language,
    ]):
        filter_info = FilterInfo(
            store_type=filters.store_type,
            category=filters.category,
            effective_date=str(filters.effective_date) if filters.effective_date else None,
            language=filters.language,
        )

    # Build metadata
    meta = Meta(
        trace_id=trace_id,
        topk=final_state.get("topk", 8),
        retrieval_attempts=counters.retrieval_attempts if counters else 1,
        generation_attempts=counters.generation_attempts if counters else 1,
        latency_ms=round(total_latency_ms, 2),
        verdict=verdict,
        filter_info=filter_info,
    )

    # Build response
    answer = ""
    follow_up_question: Optional[str] = None

    if final_response:
        answer = final_response.answer
        follow_up_question = final_response.follow_up_question

    logger.info(
        f"[{trace_id}] Chat request completed",
        extra={
            "trace_id": trace_id,
            "latency_ms": round(total_latency_ms, 2),
            "verdict": verdict.value,
            "citations_count": len(citations),
            "retrieval_attempts": counters.retrieval_attempts if counters else 1,
        },
    )

    return ChatResponse(
        answer=answer,
        citations=citations,
        follow_up_question=follow_up_question,
        meta=meta,
    )
