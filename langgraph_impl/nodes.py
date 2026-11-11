"""Agent node definitions for the LangGraph research pipeline."""

from __future__ import annotations

import logging
from typing import Any

from shared.config import Settings, get_config
from shared.formatting import print_agent_step
from shared.ollama_client import retry_with_backoff
from shared.prompts import (
    ANALYST_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    format_analyst_input,
    format_researcher_input,
    format_summarizer_input,
)
from shared.wikipedia_tool import wikipedia_research

logger = logging.getLogger(__name__)


def _build_messages(system_prompt: str, user_prompt: str) -> list[Any]:
    """Build message structure for ChatModel: supports LangChain classes or tuple format."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    except ImportError:
        return [
            ("system", system_prompt),
            ("human", user_prompt),
        ]



def get_chat_model(config: Settings | None = None) -> Any:
    """Instantiate the ChatOllama chat model integration."""
    cfg = config or get_config()
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.model_name,
            base_url=cfg.ollama_host,
            temperature=cfg.temperature,
            timeout=cfg.request_timeout,
        )
    except ImportError:
        # Fallback wrapper for environments/tests without langchain-ollama installed
        logger.debug("langchain_ollama not found; using fallback model mock.")
        return None


def researcher_node(
    state: dict[str, Any],
    config: Settings | None = None,
    llm: Any = None,
) -> dict[str, Any]:
    """Researcher agent node: gathers Wikipedia facts about the topic and outputs a brief."""
    cfg = config or get_config()
    model = llm or get_chat_model(cfg)
    topic = state.get("topic", "")

    print_agent_step("Researcher", f"Initiating Wikipedia research on topic '{topic}'...")

    tool_calls_made = 0
    llm_calls_made = 0

    # 1. Execute Wikipedia research tool
    try:
        print_agent_step("Researcher", "Querying Wikipedia knowledge base...")
        raw_facts = wikipedia_research(topic)
        tool_calls_made += 1
    except Exception as exc:
        raw_facts = f"Error querying Wikipedia: {exc}"

    # 2. Invoke LLM to synthesize the raw facts into a structured factual brief
    user_prompt = (
        f"{format_researcher_input(topic)}\n\n"
        f"--- WIKIPEDIA RAW TOOL RESULTS ---\n"
        f"{raw_facts}\n"
        f"--- END RAW RESULTS ---\n\n"
        "Synthesize these facts into the structured factual brief as specified."
    )

    if model is not None:
        try:
            def _invoke_llm() -> str:
                if hasattr(model, "invoke"):
                    msgs = _build_messages(RESEARCHER_SYSTEM_PROMPT, user_prompt)
                    resp = model.invoke(msgs)
                    return getattr(resp, "content", str(resp))
                elif callable(model):
                    return str(model(user_prompt))
                return raw_facts

            research_text = retry_with_backoff(
                _invoke_llm,
                max_retries=cfg.max_retries,
                backoff_factor=cfg.retry_backoff_factor,
            )
            llm_calls_made += 1
        except Exception as exc:
            logger.error("Researcher LLM invocation failed: %s", exc)
            research_text = f"Research synthesis fallback (facts preserved):\n\n{raw_facts}"
    else:
        research_text = f"Research Brief for '{topic}':\n\n{raw_facts}"

    print_agent_step("Researcher", "Research brief completed.", research_text[:180])

    return {
        "research_output": research_text,
        "tool_call_count": state.get("tool_call_count", 0) + tool_calls_made,
        "llm_call_count": state.get("llm_call_count", 0) + llm_calls_made,
    }


def analyst_node(
    state: dict[str, Any],
    config: Settings | None = None,
    llm: Any = None,
) -> dict[str, Any]:
    """Analyst agent node: receives research and critically analyzes findings, relationships, and gaps."""
    cfg = config or get_config()
    model = llm or get_chat_model(cfg)
    topic = state.get("topic", "")
    research_output = state.get("research_output", "")

    print_agent_step("Analyst", "Analyzing research findings, patterns, and contradictions...")

    llm_calls_made = 0
    user_prompt = format_analyst_input(topic, research_output)

    if model is not None:
        try:
            def _invoke_llm() -> str:
                if hasattr(model, "invoke"):
                    msgs = _build_messages(ANALYST_SYSTEM_PROMPT, user_prompt)
                    resp = model.invoke(msgs)
                    return getattr(resp, "content", str(resp))
                elif callable(model):
                    return str(model(user_prompt))
                return f"Analysis of {topic}"

            analysis_text = retry_with_backoff(
                _invoke_llm,
                max_retries=cfg.max_retries,
                backoff_factor=cfg.retry_backoff_factor,
            )
            llm_calls_made += 1
        except Exception as exc:
            logger.error("Analyst LLM invocation failed: %s", exc)
            analysis_text = f"Analysis unavailable due to error: {exc}"
    else:
        analysis_text = (
            f"Analytical Assessment for '{topic}':\n"
            f"1. Key Findings: Identified core principles.\n"
            f"2. Relationships: Strong correlation with computational theory.\n"
            f"3. Gaps: Requires further empirical validation."
        )

    print_agent_step("Analyst", "Analysis completed.", analysis_text[:180])

    return {
        "analysis_output": analysis_text,
        "llm_call_count": state.get("llm_call_count", 0) + llm_calls_made,
    }


def summarizer_node(
    state: dict[str, Any],
    config: Settings | None = None,
    llm: Any = None,
) -> dict[str, Any]:
    """Summarizer agent node: produces concise final summary from research and analysis."""
    cfg = config or get_config()
    model = llm or get_chat_model(cfg)
    topic = state.get("topic", "")
    research_output = state.get("research_output", "")
    analysis_output = state.get("analysis_output", "")

    print_agent_step("Summarizer", "Synthesizing executive summary...")

    llm_calls_made = 0
    user_prompt = format_summarizer_input(topic, research_output, analysis_output)

    if model is not None:
        try:
            def _invoke_llm() -> str:
                if hasattr(model, "invoke"):
                    msgs = _build_messages(SUMMARIZER_SYSTEM_PROMPT, user_prompt)
                    resp = model.invoke(msgs)
                    return getattr(resp, "content", str(resp))
                elif callable(model):
                    return str(model(user_prompt))
                return f"Executive summary of {topic}"

            summary_text = retry_with_backoff(
                _invoke_llm,
                max_retries=cfg.max_retries,
                backoff_factor=cfg.retry_backoff_factor,
            )
            llm_calls_made += 1
        except Exception as exc:
            logger.error("Summarizer LLM invocation failed: %s", exc)
            summary_text = f"Executive summary failed: {exc}"
    else:
        summary_text = (
            f"Executive Summary for '{topic}':\n\n"
            f"Overview: Comprehensive overview grounded in Wikipedia research.\n"
            f"Core Pillars: Verified architecture and analysis.\n"
            f"Outlook: Rapid evolution in local multi-agent systems."
        )

    print_agent_step("Summarizer", "Summary generated successfully.")

    return {
        "final_summary": summary_text,
        "llm_call_count": state.get("llm_call_count", 0) + llm_calls_made,
    }
