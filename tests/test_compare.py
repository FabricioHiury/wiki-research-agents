"""Unit tests for compare script execution with mocks."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from compare import main


class TestCompare(unittest.TestCase):
    @patch("compare.run_strands_pipeline")
    @patch("compare.run_langgraph_pipeline")
    @patch("sys.argv", ["compare.py", "Robotics"])
    def test_compare_main_execution(
        self, mock_langgraph: MagicMock, mock_strands: MagicMock
    ) -> None:
        mock_langgraph.return_value = {
            "framework": "LangGraph",
            "topic": "Robotics",
            "success": True,
            "final_summary": "LangGraph summary on Robotics",
            "elapsed_seconds": 1.25,
            "llm_calls": 3,
            "tool_calls": 1,
        }
        mock_strands.return_value = {
            "framework": "Strands Agents",
            "topic": "Robotics",
            "success": True,
            "final_summary": "Strands summary on Robotics",
            "elapsed_seconds": 1.10,
            "llm_calls": 3,
            "tool_calls": 1,
        }

        # Should complete without error
        main()
        mock_langgraph.assert_called_once()
        mock_strands.assert_called_once()


if __name__ == "__main__":
    unittest.main()
