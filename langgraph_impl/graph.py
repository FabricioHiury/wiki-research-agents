"""Graph compilation and execution for the LangGraph research pipeline."""

from __future__ import annotations

import time
from typing import Any, Callable

from langgraph_impl.nodes import (
    analyst_node,
    get_chat_model,
    researcher_node,
    summarizer_node,
)
from langgraph_impl.state import PipelineGraphState
from shared.config import Settings, get_config
from shared.ollama_client import check_ollama_status


class SequentialFallbackGraph:
    """Fallback runner providing identical execution semantics when langgraph is not installed."""

    def __init__(self, nodes: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]]) -> None:
        self.nodes = nodes

    def invoke(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        state = dict(initial_state)
        for _name, node_func in self.nodes:
            output = node_func(state)
            state.update(output)
        return state


def build_research_graph(config: Settings | None = None, custom_llm: Any = None) -> Any:
    """Build and compile the LangGraph StateGraph pipeline.

    Defines explicit nodes (researcher, analyst, summarizer) and linear edges:
    START -> researcher -> analyst -> summarizer -> END.
    """
    cfg = config or get_config()
    llm = custom_llm or get_chat_model(cfg)

    def bound_researcher(state: dict[str, Any]) -> dict[str, Any]:
        return researcher_node(state, config=cfg, llm=llm)

    def bound_analyst(state: dict[str, Any]) -> dict[str, Any]:
        return analyst_node(state, config=cfg, llm=llm)

    def bound_summarizer(state: dict[str, Any]) -> dict[str, Any]:
        return summarizer_node(state, config=cfg, llm=llm)

    try:
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(PipelineGraphState)
        builder.add_node("researcher", bound_researcher)
        builder.add_node("analyst", bound_analyst)
        builder.add_node("summarizer", bound_summarizer)

        builder.add_edge(START, "researcher")
        builder.add_edge("researcher", "analyst")
        builder.add_edge("analyst", "summarizer")
        builder.add_edge("summarizer", END)

        return builder.compile()
    except ImportError:
        # Fallback to pure-Python sequential graph
        return SequentialFallbackGraph([
            ("researcher", bound_researcher),
            ("analyst", bound_analyst),
            ("summarizer", bound_summarizer),
        ])


def run_langgraph_pipeline(
    topic: str,
    config: Settings | None = None,
    check_server: bool = True,
    custom_llm: Any = None,
) -> dict[str, Any]:
    """Execute the complete LangGraph research pipeline for a topic.

    Args:
        topic: The user's query or research subject.
        config: Optional application settings.
        check_server: Whether to perform pre-flight Ollama healthcheck.
        custom_llm: Optional mock LLM for testing.

    Returns:
        Dictionary containing summary, intermediate outputs, and execution metrics.
    """
    cfg = config or get_config()

    # Pre-flight check
    if check_server and custom_llm is None:
        ok, msg = check_ollama_status(host=cfg.ollama_host, model_name=cfg.model_name)
        if not ok:
            return {
                "framework": "LangGraph",
                "topic": topic,
                "success": False,
                "error": msg,
                "final_summary": f"Execution halted: {msg}",
                "research_output": "",
                "analysis_output": "",
                "elapsed_seconds": 0.0,
                "llm_calls": 0,
                "tool_calls": 0,
            }

    app = build_research_graph(config=cfg, custom_llm=custom_llm)

    initial_state: PipelineGraphState = {
        "topic": topic,
        "research_output": "",
        "analysis_output": "",
        "final_summary": "",
        "llm_call_count": 0,
        "tool_call_count": 0,
        "errors": [],
    }

    start_time = time.perf_counter()
    try:
        final_state = app.invoke(initial_state)
        elapsed = time.perf_counter() - start_time

        return {
            "framework": "LangGraph",
            "topic": topic,
            "success": True,
            "error": None,
            "final_summary": final_state.get("final_summary", ""),
            "research_output": final_state.get("research_output", ""),
            "analysis_output": final_state.get("analysis_output", ""),
            "elapsed_seconds": elapsed,
            "llm_calls": final_state.get("llm_call_count", 0),
            "tool_calls": final_state.get("tool_call_count", 0),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return {
            "framework": "LangGraph",
            "topic": topic,
            "success": False,
            "error": str(exc),
            "final_summary": f"Pipeline failure: {exc}",
            "research_output": "",
            "analysis_output": "",
            "elapsed_seconds": elapsed,
            "llm_calls": 0,
            "tool_calls": 0,
        }

