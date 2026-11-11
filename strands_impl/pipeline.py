"""Sequential multi-agent orchestration using Strands Agents."""

from __future__ import annotations

import logging
import time
from typing import Any

from shared.config import Settings, get_config
from shared.formatting import print_agent_step
from shared.ollama_client import check_ollama_status, retry_with_backoff
from shared.prompts import (
    format_analyst_input,
    format_researcher_input,
    format_summarizer_input,
)
from shared.wikipedia_tool import wikipedia_research
from strands_impl.agents import (
    create_analyst_agent,
    create_researcher_agent,
    create_summarizer_agent,
    get_strands_model,
)

logger = logging.getLogger(__name__)


def run_strands_pipeline(
    topic: str,
    config: Settings | None = None,
    check_server: bool = True,
    custom_model: Any = None,
) -> dict[str, Any]:
    """Execute the sequential multi-agent research pipeline using Strands Agents.

    Pipeline sequence:
      1. Researcher (uses Wikipedia tool to produce factual brief)
      2. Analyst (evaluates key points, patterns, and gaps)
      3. Summarizer (synthesizes concise executive summary)

    Args:
        topic: The subject to research.
        config: Optional application settings.
        check_server: Whether to perform pre-flight Ollama healthcheck.
        custom_model: Optional model/mock for testing.

    Returns:
        Structured result dictionary with timings and call metrics.
    """
    cfg = config or get_config()

    # Pre-flight server check
    if check_server and custom_model is None:
        ok, msg = check_ollama_status(host=cfg.ollama_host, model_name=cfg.model_name)
        if not ok:
            return {
                "framework": "Strands Agents",
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

    model = custom_model or get_strands_model(cfg)

    # Initialize specialized agents
    researcher = create_researcher_agent(model, cfg)
    analyst = create_analyst_agent(model, cfg)
    summarizer = create_summarizer_agent(model, cfg)

    llm_calls = 0
    tool_calls = 0
    start_time = time.perf_counter()

    try:
        # ----------------------------------------------------------------------
        # Step 1: Researcher Agent
        # ----------------------------------------------------------------------
        print_agent_step("Researcher", f"Researching topic '{topic}' using Wikipedia tool...")

        # In local Ollama, ensure tool information is explicitly available to Researcher
        tool_facts = wikipedia_research(topic)
        tool_calls += 1

        research_prompt = (
            f"{format_researcher_input(topic)}\n\n"
            f"--- WIKIPEDIA RAW TOOL RESULTS ---\n"
            f"{tool_facts}\n"
            f"--- END RAW RESULTS ---\n\n"
            "Synthesize these facts into the structured factual brief."
        )

        def _run_researcher() -> str:
            res = researcher(research_prompt)
            return str(getattr(res, "text", res))

        research_output = retry_with_backoff(
            _run_researcher,
            max_retries=cfg.max_retries,
            backoff_factor=cfg.retry_backoff_factor,
        )
        llm_calls += 1
        print_agent_step("Researcher", "Factual research brief compiled.", research_output[:180])

        # ----------------------------------------------------------------------
        # Step 2: Analyst Agent
        # ----------------------------------------------------------------------
        print_agent_step("Analyst", "Analyzing research findings and implications...")
        analyst_prompt = format_analyst_input(topic, research_output)

        def _run_analyst() -> str:
            res = analyst(analyst_prompt)
            return str(getattr(res, "text", res))

        analysis_output = retry_with_backoff(
            _run_analyst,
            max_retries=cfg.max_retries,
            backoff_factor=cfg.retry_backoff_factor,
        )
        llm_calls += 1
        print_agent_step("Analyst", "Analytical assessment complete.", analysis_output[:180])

        # ----------------------------------------------------------------------
        # Step 3: Summarizer Agent
        # ----------------------------------------------------------------------
        print_agent_step("Summarizer", "Crafting final executive synthesis...")
        summarizer_prompt = format_summarizer_input(topic, research_output, analysis_output)

        def _run_summarizer() -> str:
            res = summarizer(summarizer_prompt)
            return str(getattr(res, "text", res))

        final_summary = retry_with_backoff(
            _run_summarizer,
            max_retries=cfg.max_retries,
            backoff_factor=cfg.retry_backoff_factor,
        )
        llm_calls += 1
        print_agent_step("Summarizer", "Executive summary finalized.")

        elapsed = time.perf_counter() - start_time

        return {
            "framework": "Strands Agents",
            "topic": topic,
            "success": True,
            "error": None,
            "final_summary": final_summary,
            "research_output": research_output,
            "analysis_output": analysis_output,
            "elapsed_seconds": elapsed,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.error("Strands pipeline execution error: %s", exc)
        return {
            "framework": "Strands Agents",
            "topic": topic,
            "success": False,
            "error": str(exc),
            "final_summary": f"Pipeline aborted due to error: {exc}",
            "research_output": "",
            "analysis_output": "",
            "elapsed_seconds": elapsed,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
        }

