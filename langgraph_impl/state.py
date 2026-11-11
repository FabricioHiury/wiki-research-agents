"""State definitions for the LangGraph multi-agent pipeline."""

from __future__ import annotations

from typing import TypedDict


class PipelineGraphState(TypedDict, total=False):
    """Shared typed state flowing across all nodes in the LangGraph pipeline."""

    # Input topic
    topic: str

    # Artifacts generated sequentially by each agent
    research_output: str
    analysis_output: str
    final_summary: str

    # Execution telemetry & accounting
    llm_call_count: int
    tool_call_count: int
    errors: list[str]
