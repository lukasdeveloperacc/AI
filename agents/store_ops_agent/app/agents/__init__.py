"""LangGraph agents package for StoreOps Agent.

This package provides:
- Agent state schema (AgentState, create_initial_state)
- Graph workflow (run_agent, run_agent_with_streaming)
- Individual nodes (parse_classify, retrieve, generate, grounding_check, finalize)
"""

from app.agents.graph import (
    build_agent_graph,
    get_agent_graph,
    run_agent,
    run_agent_with_streaming,
)
from app.agents.nodes import (
    finalize_node,
    generate_node,
    grounding_check_node,
    parse_classify_node,
    retrieve_node,
)
from app.agents.nodes.retrieve import VectorStoreManager
from app.agents.state import (
    AgentCounters,
    AgentFilters,
    AgentMetrics,
    AgentState,
    DraftAnswer,
    FinalResponse,
    GroundingCheckResult,
    GroundingVerdict,
    Intent,
    ParsedQuery,
    RetrievalResult,
    create_initial_state,
)

__all__ = [
    # State types
    "AgentState",
    "AgentCounters",
    "AgentFilters",
    "AgentMetrics",
    "DraftAnswer",
    "FinalResponse",
    "GroundingCheckResult",
    "GroundingVerdict",
    "Intent",
    "ParsedQuery",
    "RetrievalResult",
    "create_initial_state",
    # Graph
    "build_agent_graph",
    "get_agent_graph",
    "run_agent",
    "run_agent_with_streaming",
    # Nodes
    "parse_classify_node",
    "retrieve_node",
    "generate_node",
    "grounding_check_node",
    "finalize_node",
    # Utils
    "VectorStoreManager",
]
