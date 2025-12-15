"""Parse and Classify Node for StoreOps Agent.

This node:
1. Normalizes the user query
2. Classifies intent (policy_question, procedure_question, general_question, out_of_scope)
3. Extracts metadata hints for filtering (category, store_type, effective_date)
4. Returns normalized_query, intent, confidence, and filters
"""

import logging
import re
import time
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.state import (
    AgentFilters,
    AgentState,
    Intent,
    ParsedQuery,
)

logger = logging.getLogger(__name__)


class QueryClassification(BaseModel):
    """Schema for LLM query classification response."""

    normalized_query: str = Field(
        description="Normalized query with typos corrected and key terms preserved"
    )
    intent: str = Field(
        description="Intent classification: policy_question, procedure_question, general_question, or out_of_scope"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for classification"
    )
    category: str | None = Field(
        default=None,
        description="Detected category: refund, promo, inventory, cs, operation, hr",
    )
    store_type: str | None = Field(
        default=None,
        description="Detected store type: cafe, convenience, apparel, restaurant, retail",
    )
    reasoning: str = Field(description="Brief reasoning for the classification")


CLASSIFICATION_SYSTEM_PROMPT = """You are a query classifier for a store operations Q&A system.

Your tasks:
1. Normalize the query: fix typos, remove unnecessary words, but preserve key terms
2. Classify the intent:
   - policy_question: Questions about rules, regulations, policies (환불 규정, 근무 시간 등)
   - procedure_question: Questions about how to do something (절차, 방법, 프로세스)
   - general_question: General information queries
   - out_of_scope: Questions unrelated to store operations (weather, personal, etc.)
3. Extract metadata hints if present:
   - category: refund(환불), promo(프로모션), inventory(재고), cs(고객서비스), operation(운영), hr(인사)
   - store_type: cafe(카페), convenience(편의점), apparel(의류), restaurant(레스토랑), retail(소매)

Examples:
- "카페에서 환불 처리 어떻게 해요?" -> intent: procedure_question, category: refund, store_type: cafe
- "편의점 야간 수당 규정이 뭐예요?" -> intent: policy_question, category: hr, store_type: convenience
- "오늘 날씨 어때?" -> intent: out_of_scope

Respond in JSON format matching the schema."""


async def parse_classify_node(state: AgentState) -> dict[str, Any]:
    """Parse and classify the user query.

    Args:
        state: Current agent state with question.

    Returns:
        Dictionary with parsed query and updated filters.
    """
    start_time = time.time()
    question = state["question"]
    existing_filters = state.get("filters", AgentFilters())

    try:
        # Use LLM for classification
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(QueryClassification)

        result: QueryClassification = await structured_llm.ainvoke(
            [
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )

        # Map intent string to enum
        intent_map = {
            "policy_question": Intent.POLICY_QUESTION,
            "procedure_question": Intent.PROCEDURE_QUESTION,
            "general_question": Intent.GENERAL_QUESTION,
            "out_of_scope": Intent.OUT_OF_SCOPE,
        }
        intent = intent_map.get(result.intent.lower(), Intent.GENERAL_QUESTION)

        # Create parsed query
        parsed = ParsedQuery(
            normalized_query=result.normalized_query,
            intent=intent,
            confidence=result.confidence,
            extracted_entities={
                "category": result.category,
                "store_type": result.store_type,
                "reasoning": result.reasoning,
            },
        )

        # Update filters with extracted values (only if not already set)
        updated_filters = AgentFilters(
            category=existing_filters.category or result.category,
            store_type=existing_filters.store_type or result.store_type,
            effective_date=existing_filters.effective_date,
            language=existing_filters.language,
        )

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Parse/Classify completed: intent={intent.value}, "
            f"confidence={result.confidence:.2f}, latency={latency_ms:.1f}ms"
        )

        # Update metrics
        current_metrics = state.get("metrics")
        if current_metrics:
            current_metrics.parse_latency_ms = latency_ms

        return {
            "parsed": parsed,
            "filters": updated_filters,
            "metrics": current_metrics,
        }

    except Exception as e:
        logger.error(f"Parse/Classify failed: {e}")
        latency_ms = (time.time() - start_time) * 1000

        # Fallback to simple normalization
        parsed = ParsedQuery(
            normalized_query=_simple_normalize(question),
            intent=Intent.GENERAL_QUESTION,
            confidence=0.5,
            extracted_entities={"error": str(e)},
        )

        errors = state.get("errors", [])
        errors.append(f"Parse/Classify error: {e}")

        return {
            "parsed": parsed,
            "filters": existing_filters,
            "errors": errors,
        }


def _simple_normalize(query: str) -> str:
    """Simple query normalization fallback.

    Args:
        query: Original query string.

    Returns:
        Normalized query string.
    """
    # Remove extra whitespace
    normalized = " ".join(query.split())
    # Remove special characters except Korean, alphanumeric, and basic punctuation
    normalized = re.sub(r"[^\w\s가-힣.,?!]", "", normalized)
    return normalized.strip()
