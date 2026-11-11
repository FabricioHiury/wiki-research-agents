"""Unit tests for the Wikipedia research tool with mocked HTTP requests."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from shared.wikipedia_tool import (
    get_wikipedia_page_summary,
    search_wikipedia_titles,
    wikipedia_research,
)


class TestWikipediaTool(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_search_wikipedia_titles_success(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        # OpenSearch API payload format: [query, [titles], [descriptions], [urls]]
        payload = [
            "Python",
            ["Python (programming language)", "Python (genus)", "Monty Python"],
            ["", "", ""],
            ["", "", ""],
        ]
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        titles = search_wikipedia_titles("Python", limit=3)
        self.assertEqual(len(titles), 3)
        self.assertEqual(titles[0], "Python (programming language)")

    def test_search_wikipedia_titles_empty_query(self) -> None:
        self.assertEqual(search_wikipedia_titles(""), [])
        self.assertEqual(search_wikipedia_titles("   "), [])

    @patch("urllib.request.urlopen")
    def test_get_wikipedia_page_summary_success(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        payload = {
            "title": "Artificial intelligence",
            "type": "standard",
            "extract": "Artificial intelligence (AI) is the intelligence of machines or software.",
            "content_urls": {
                "desktop": {
                    "page": "https://en.wikipedia.org/wiki/Artificial_intelligence"
                }
            },
        }
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        result = get_wikipedia_page_summary("Artificial intelligence")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["title"], "Artificial intelligence")
        self.assertIn("Artificial intelligence (AI)", result["extract"])
        self.assertEqual(
            result["url"], "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )

    @patch("urllib.request.urlopen")
    def test_get_wikipedia_page_summary_disambiguation(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        payload = {
            "title": "Mercury",
            "type": "disambiguation",
            "extract": "Mercury most often refers to: Mercury (element), or Mercury (planet).",
            "content_urls": {
                "desktop": {"page": "https://en.wikipedia.org/wiki/Mercury"}
            },
        }
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        result = get_wikipedia_page_summary("Mercury")
        self.assertEqual(result["status"], "disambiguation")
        self.assertIn("disambiguation", result["message"])

    @patch("urllib.request.urlopen")
    def test_get_wikipedia_page_summary_not_found(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        payload = {
            "type": "https://mediawiki.org/wiki/HyperSwitch/errors/not_found",
            "title": "Not found",
        }
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        result = get_wikipedia_page_summary("NonExistentRandomTopic123456")
        self.assertEqual(result["status"], "not_found")
        self.assertIn("does not exist", result["message"])

    @patch("shared.wikipedia_tool.search_wikipedia_titles")
    @patch("shared.wikipedia_tool.get_wikipedia_page_summary")
    def test_wikipedia_research_success_flow(
        self, mock_get_summary: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_search.return_value = ["Quantum computing"]
        mock_get_summary.return_value = {
            "status": "ok",
            "title": "Quantum computing",
            "extract": "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics.",
            "url": "https://en.wikipedia.org/wiki/Quantum_computing",
        }

        research_text = wikipedia_research("Quantum computing")
        self.assertIn("Wikipedia Article: Quantum computing", research_text)
        self.assertIn("https://en.wikipedia.org/wiki/Quantum_computing", research_text)
        self.assertIn("laws of quantum mechanics", research_text)

    def test_wikipedia_research_empty_query(self) -> None:
        res = wikipedia_research("")
        self.assertEqual(res, "No topic provided for Wikipedia research.")


if __name__ == "__main__":
    unittest.main()

