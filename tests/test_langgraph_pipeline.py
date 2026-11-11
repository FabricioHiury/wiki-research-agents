"""Unit tests for the LangGraph pipeline with mocked LLM and Wikipedia tool."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langgraph_impl.graph import build_research_graph, run_langgraph_pipeline
from langgraph_impl.nodes import analyst_node, researcher_node, summarizer_node
from shared.config import Settings


class MockLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class TestLangGraphPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Settings(
            ollama_host="http://localhost:11434",
            model_name="llama3.1:8b",
            temperature=0.0,
            max_retries=1,
            retry_backoff_factor=1.0,
        )

    @patch("langgraph_impl.nodes.wikipedia_research")
    def test_researcher_node_invokes_tool_and_llm(
        self, mock_wiki: MagicMock
    ) -> None:
        mock_wiki.return_value = "Wikipedia factual excerpt about Neural Networks."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "Synthesized Research Brief on Neural Networks."
        )

        initial_state = {
            "topic": "Neural Networks",
            "llm_call_count": 0,
            "tool_call_count": 0,
        }
        res = researcher_node(initial_state, config=self.config, llm=mock_llm)

        mock_wiki.assert_called_once_with("Neural Networks")
        self.assertIn("Synthesized Research Brief", res["research_output"])
        self.assertEqual(res["tool_call_count"], 1)
        self.assertEqual(res["llm_call_count"], 1)

    def test_analyst_node_invokes_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "Critical Analysis: Key findings and gaps identified."
        )

        state = {
            "topic": "Neural Networks",
            "research_output": "Factual brief",
            "llm_call_count": 1,
        }
        res = analyst_node(state, config=self.config, llm=mock_llm)

        self.assertIn("Critical Analysis", res["analysis_output"])
        self.assertEqual(res["llm_call_count"], 2)

    def test_summarizer_node_invokes_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "Final Executive Summary: Modern Neural Networks."
        )

        state = {
            "topic": "Neural Networks",
            "research_output": "Factual brief",
            "analysis_output": "Critical analysis",
            "llm_call_count": 2,
        }
        res = summarizer_node(state, config=self.config, llm=mock_llm)

        self.assertIn("Final Executive Summary", res["final_summary"])
        self.assertEqual(res["llm_call_count"], 3)

    @patch("langgraph_impl.nodes.wikipedia_research")
    def test_run_langgraph_pipeline_end_to_end_mocked(
        self, mock_wiki: MagicMock
    ) -> None:
        mock_wiki.return_value = "Facts about Quantum Computing."
        mock_llm = MagicMock()

        def llm_side_effect(messages: list) -> MockLLMResponse:
            prompt_str = str(messages)
            if "Fact Researcher" in prompt_str or "WIKIPEDIA RAW" in prompt_str:
                return MockLLMResponse("Research Brief: Quantum Computing facts.")
            elif "Intelligence Analyst" in prompt_str:
                return MockLLMResponse("Analysis: Superposition and entanglement.")
            return MockLLMResponse("Executive Summary: Quantum Computing Revolution.")

        mock_llm.invoke.side_effect = llm_side_effect

        result = run_langgraph_pipeline(
            topic="Quantum Computing",
            config=self.config,
            check_server=False,
            custom_llm=mock_llm,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["framework"], "LangGraph")
        self.assertIn("Executive Summary", result["final_summary"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["llm_calls"], 3)
        self.assertGreater(result["elapsed_seconds"], 0.0)

    @patch("langgraph_impl.graph.check_ollama_status")
    def test_run_langgraph_pipeline_server_unreachable(
        self, mock_check: MagicMock
    ) -> None:
        mock_check.return_value = (False, "Ollama is not reachable at http://localhost:11434")

        result = run_langgraph_pipeline(
            topic="Robotics",
            config=self.config,
            check_server=True,
            custom_llm=None,
        )

        self.assertFalse(result["success"])
        self.assertIn("not reachable", result["error"])
        self.assertEqual(result["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()

