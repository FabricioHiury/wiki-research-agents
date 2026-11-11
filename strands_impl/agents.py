"""Agent and tool definitions using Strands Agents SDK."""

from __future__ import annotations

import logging
from typing import Any, Callable

from shared.config import Settings, get_config
from shared.prompts import (
    ANALYST_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
)
from shared.wikipedia_tool import wikipedia_research

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Wikipedia Tool definition for Strands Agents
# ---------------------------------------------------------------------------
def _define_wikipedia_tool() -> Any:
    """Register wikipedia_research as a Strands agent tool with docstring schema."""
    try:
        from strands import tool

        @tool
        def wikipedia_tool(query: str) -> str:
            """Query Wikipedia to research factual information about a topic or concept.

            Args:
                query: The topic, person, system, or scientific concept to search for.

            Returns:
                Factual excerpt and reference details from Wikipedia.
            """
            return wikipedia_research(query)

        return wikipedia_tool
    except ImportError:
        def wikipedia_tool(query: str) -> str:
            return wikipedia_research(query)

        return wikipedia_tool


wikipedia_tool = _define_wikipedia_tool()


# ---------------------------------------------------------------------------
# 2. Strands Ollama Model Provider Factory
# ---------------------------------------------------------------------------
def get_strands_model(config: Settings | None = None) -> Any:
    """Instantiate the native Strands Ollama model provider."""
    cfg = config or get_config()
    try:
        from strands.models.ollama import OllamaModel

        return OllamaModel(
            host=cfg.ollama_host,
            model_id=cfg.model_name,
            temperature=cfg.temperature,
            timeout=cfg.request_timeout,
        )
    except ImportError:
        logger.debug("strands.models.ollama not installed; using fallback model.")
        return None


# ---------------------------------------------------------------------------
# 3. Fallback Agent implementation for tests/environments without strands-agents
# ---------------------------------------------------------------------------
class FallbackStrandsAgent:
    """Lightweight fallback matching the Strands Agent callable interface."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Any] | None = None,
        model: Any = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model

    def __call__(self, prompt: str) -> str:
        if self.model and callable(self.model):
            return str(self.model(prompt))
        if hasattr(self.model, "invoke"):
            resp = self.model.invoke(prompt)
            return getattr(resp, "content", str(resp))
        return f"[{self.name} Response] Processed: {prompt[:80]}"


# ---------------------------------------------------------------------------
# 4. Agent Factory Functions
# ---------------------------------------------------------------------------
def create_researcher_agent(
    model: Any,
    config: Settings | None = None,
) -> Any:
    """Create the Researcher Agent equipped with the Wikipedia tool."""
    try:
        from strands import Agent

        return Agent(
            model=model,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            tools=[wikipedia_tool],
        )
    except ImportError:
        return FallbackStrandsAgent(
            name="Researcher",
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            tools=[wikipedia_tool],
            model=model,
        )


def create_analyst_agent(
    model: Any,
    config: Settings | None = None,
) -> Any:
    """Create the Analyst Agent focused on strategic evaluation."""
    try:
        from strands import Agent

        return Agent(
            model=model,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            tools=[],
        )
    except ImportError:
        return FallbackStrandsAgent(
            name="Analyst",
            system_prompt=ANALYST_SYSTEM_PROMPT,
            tools=[],
            model=model,
        )


def create_summarizer_agent(
    model: Any,
    config: Settings | None = None,
) -> Any:
    """Create the Summarizer Agent for executive synthesis."""
    try:
        from strands import Agent

        return Agent(
            model=model,
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            tools=[],
        )
    except ImportError:
        return FallbackStrandsAgent(
            name="Summarizer",
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            tools=[],
            model=model,
        )

