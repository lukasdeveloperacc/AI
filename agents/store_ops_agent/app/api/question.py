"""Question answering API endpoints."""

from fastapi import APIRouter, HTTPException

from app.core.citation_formatter import CitationFormatter
from app.core.tracing import trace_request
from app.models.schemas import AnswerResponse, Citation, QuestionRequest, Verdict

router = APIRouter(prefix="/api/v1", tags=["questions"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """Ask a question and get an answer with citations.

    This endpoint receives a question, retrieves relevant documents,
    and generates an answer with source citations.

    Args:
        request: The question request containing the user's question.

    Returns:
        AnswerResponse with the answer, citations, and metadata.

    Raises:
        HTTPException: If processing fails.
    """
    with trace_request(topk=request.topk) as tracer:
        try:
            # Record retrieval attempt
            tracer.record_retrieval_attempt()

            # TODO: Implement actual document retrieval and LLM response
            # For now, return a mock response demonstrating the schema

            # Mock citations - will be replaced with actual retrieval results
            mock_citations = [
                Citation(
                    doc_id="doc_001",
                    title="매장 운영 매뉴얼",
                    chunk_id="chunk_003",
                    snippet="매장 운영에 관한 기본 지침이 포함되어 있습니다.",
                    score=0.92,
                ),
                Citation(
                    doc_id="doc_002",
                    title="직원 근무 규정",
                    chunk_id="chunk_012",
                    snippet="직원 근무 시간 및 휴무에 관한 규정입니다.",
                    score=0.85,
                ),
            ]

            # Filter and sort citations
            formatter = CitationFormatter()
            citations = formatter.sort_by_score(mock_citations)
            citations = formatter.filter_by_score(citations, min_score=0.5)

            # Determine verdict based on citations
            tracer.determine_verdict(
                citations_count=len(citations),
                min_citations=2,
                has_answer=True,
            )

            # Mock answer - will be replaced with LLM generated response
            mock_answer = (
                f"'{request.question}'에 대한 답변입니다. "
                "관련 문서를 검색하여 정확한 정보를 제공하겠습니다. "
                "자세한 내용은 아래 근거 자료를 참고해 주세요."
            )

            # Generate follow-up questions (optional)
            follow_up_questions = [
                "추가로 궁금한 점이 있으신가요?",
                "다른 매장 운영 관련 질문이 있으신가요?",
            ]

            return AnswerResponse(
                answer=mock_answer,
                citations=citations,
                follow_up_questions=follow_up_questions,
                meta=tracer.to_meta(),
            )

        except Exception as e:
            tracer.set_verdict(Verdict.NOT_FOUND)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process question: {str(e)}",
            ) from e


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for the question API.

    Returns:
        Health status dictionary.
    """
    return {"status": "healthy", "service": "question-api"}
