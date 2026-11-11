"""Unit tests for the Strands Agents pipeline with mocked LLM and tools."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from shared.config import Settings
from strands_impl.agents import (
    create_analyst_agent,
    create_researcher_agent,
    create_summarizer_agent,
    wikipedia_tool,
)
from strands_impl.pipeline import run_strands_pipeline


class TestStrandsPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Settings(
            ollama_host="http://localhost:11434",
            model_name="llama3.1:8b",
            temperature=0.0,
            max_retries=1,
            retry_backoff_factor=1.0,
        )

    def test_agent_factories_initialization(self) -> None:
        mock_model = MagicMock()
        researcher = create_researcher_agent(mock_model, self.config)
        analyst = create_analyst_agent(mock_model, self.config)
        summarizer = create_summarizer_agent(mock_model, self.config)

        self.assertIsNotNone(researcher)
        self.assertIsNotNone(analyst)
        self.assertIsNotNone(summarizer)

    @patch("strands_impl.pipeline.wikipedia_research")
    def test_run_strands_pipeline_success_mocked(
        self, mock_wiki: MagicMock
    ) -> None:
        mock_wiki.return_value = "Factual content about Reinforcement Learning."

        # Mock model callable
        def mock_llm_callable(prompt: str) -> str:
            if "WIKIPEDIA RAW TOOL RESULTS" in prompt:
                return "Strands Research Brief: RL foundations."
            elif "RESEARCH BRIEF" in prompt:
                return "Strands Analysis: Exploration vs Exploitation trade-offs."
            elif "ANALYTICAL EVALUATION" in prompt:
                return "Strands Executive Summary: RL Applications."
            return "Generic agent response."

        result = run_strands_pipeline(
            topic="Reinforcement Learning",
            config=self.config,
            check_server=False,
            custom_model=mock_llm_callable,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["framework"], "Strands Agents")
        self.assertIn("Strands Executive Summary", result["final_summary"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["llm_calls"], 3)
        self.assertGreater(result["elapsed_seconds"], 0.0)

    @patch("strands_impl.pipeline.check_ollama_status")
    def test_run_strands_pipeline_server_unreachable(
        self, mock_check: MagicMock
    ) -> None:
        mock_check.return_value = (False, "Ollama is not reachable at http://localhost:11434")

        result = run_strands_pipeline(
            topic="Distributed Systems",
            config=self.config,
            check_server=True,
            custom_model=None,
        )

        self.assertFalse(result["success"])
        self.assertIn("not reachable", result["error"])
        self.assertEqual(result["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()

