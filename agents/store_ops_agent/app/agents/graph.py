"""LangGraph workflow definition for StoreOps Agent.

This module defines the agent graph with:
- Nodes: parse_classify -> retrieve -> generate -> grounding_check -> finalize
- Conditional routing: retry retrieve if grounding insufficient and attempts < 2
"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    finalize_node,
    generate_node,
    grounding_check_node,
    parse_classify_node,
    retrieve_node,
)
from app.agents.state import (
    AgentState,
    GroundingVerdict,
    create_initial_state,
)

logger = logging.getLogger(__name__)

# Maximum retrieval attempts before giving up
MAX_RETRIEVAL_ATTEMPTS = 2


def should_retry_retrieval(state: AgentState) -> Literal["retrieve", "finalize"]:
    """Determine whether to retry retrieval or proceed to finalize.

    Routing logic:
    - If grounding verdict is 'insufficient' AND retrieval_attempts < MAX:
      -> retry retrieve with relaxed filters
    - Otherwise:
      -> proceed to finalize

    Args:
        state: Current agent state.

    Returns:
        Next node name: 'retrieve' or 'finalize'.
    """
    grounding = state.get("grounding")
    counters = state.get("counters")

    if grounding is None:
        logger.warning("No grounding result, proceeding to finalize")
        return "finalize"

    # Check if we should retry
    should_retry = (
        grounding.verdict == GroundingVerdict.INSUFFICIENT
        and grounding.recommended_action == "retry_retrieval"
        and counters is not None
        and counters.retrieval_attempts < MAX_RETRIEVAL_ATTEMPTS
    )

    if should_retry:
        logger.info(
            f"Retrying retrieval (attempt {counters.retrieval_attempts + 1}/{MAX_RETRIEVAL_ATTEMPTS})"
        )
        return "retrieve"

    return "finalize"


def build_agent_graph() -> StateGraph:
    """Build the LangGraph workflow for StoreOps Agent.

    Graph structure:
    ```
    START -> parse_classify -> retrieve -> generate -> grounding_check
                                  ^                         |
                                  |                         v
                                  +---- (if insufficient) --+
                                                            |
                                                            v
                                                        finalize -> END
    ```

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create graph with state schema
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("parse_classify", parse_classify_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("grounding_check", grounding_check_node)
    graph_builder.add_node("finalize", finalize_node)

    # Add edges
    graph_builder.add_edge(START, "parse_classify")
    graph_builder.add_edge("parse_classify", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "grounding_check")

    # Conditional edge from grounding_check
    graph_builder.add_conditional_edges(
        "grounding_check",
        should_retry_retrieval,
        {
            "retrieve": "retrieve",
            "finalize": "finalize",
        },
    )

    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()


# Global compiled graph instance
_agent_graph = None


def get_agent_graph() -> StateGraph:
    """Get or create the agent graph singleton.

    Returns:
        Compiled agent graph.
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


async def run_agent(
    question: str,
    trace_id: str,
    topk: int = 8,
    store_type: str | None = None,
    category: str | None = None,
    effective_date: str | None = None,
    language: str | None = None,
) -> AgentState:
    """Run the agent workflow with a question.

    Args:
        question: User's question.
        trace_id: Unique trace ID for the request.
        topk: Number of documents to retrieve.
        store_type: Optional store type filter.
        category: Optional category filter.
        effective_date: Optional effective date filter (YYYY-MM-DD).
        language: Optional language filter.

    Returns:
        Final agent state with response.
    """
    # Create initial state
    initial_state = create_initial_state(
        question=question,
        trace_id=trace_id,
        topk=topk,
        store_type=store_type,
        category=category,
        effective_date=effective_date,
        language=language,
    )

    # Get compiled graph
    graph = get_agent_graph()

    # Run graph
    logger.info(f"Starting agent run for trace_id={trace_id}")
    final_state = await graph.ainvoke(initial_state)
    logger.info(f"Agent run completed for trace_id={trace_id}")

    return final_state


async def run_agent_with_streaming(
    question: str,
    trace_id: str,
    topk: int = 8,
    store_type: str | None = None,
    category: str | None = None,
    effective_date: str | None = None,
    language: str | None = None,
):
    """Run the agent workflow with streaming output.

    Yields state updates as the graph executes.

    Args:
        question: User's question.
        trace_id: Unique trace ID for the request.
        topk: Number of documents to retrieve.
        store_type: Optional store type filter.
        category: Optional category filter.
        effective_date: Optional effective date filter (YYYY-MM-DD).
        language: Optional language filter.

    Yields:
        Tuples of (node_name, state_update) for each node execution.
    """
    # Create initial state
    initial_state = create_initial_state(
        question=question,
        trace_id=trace_id,
        topk=topk,
        store_type=store_type,
        category=category,
        effective_date=effective_date,
        language=language,
    )

    # Get compiled graph
    graph = get_agent_graph()

    # Stream graph execution
    logger.info(f"Starting streaming agent run for trace_id={trace_id}")
    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        yield chunk
    logger.info(f"Streaming agent run completed for trace_id={trace_id}")
