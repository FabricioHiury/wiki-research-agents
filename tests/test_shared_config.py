"""Unit tests for configuration and Ollama client health check and retry."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared.config import Settings, get_config, load_dotenv
from shared.ollama_client import check_ollama_status, retry_with_backoff


class TestConfig(unittest.TestCase):
    def test_default_settings(self) -> None:
        settings = Settings()
        self.assertEqual(settings.ollama_host, "http://localhost:11434")
        self.assertEqual(settings.model_name, "llama3.1:8b")
        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.max_tool_calls, 3)

    def test_load_dotenv_parses_file(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as tf:
            tf.write("# Comment\n")
            tf.write("CUSTOM_TEST_VAR=unit_test_val\n")
            tf.write("OLLAMA_HOST=http://testhost:11434/\n")
            tf_path = Path(tf.name)

        try:
            with patch.dict("os.environ", {}, clear=True):
                load_dotenv(tf_path)
                settings = Settings.from_env(tf_path)
                self.assertEqual(settings.ollama_host, "http://testhost:11434")
        finally:
            tf_path.unlink(missing_ok=True)

    def test_get_config_singleton(self) -> None:
        c1 = get_config(reload=True)
        c2 = get_config(reload=False)
        self.assertIs(c1, c2)


class TestOllamaClient(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_check_ollama_status_success(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.read.return_value = json.dumps(
            {"models": [{"name": "llama3.1:8b"}]}
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        ok, msg = check_ollama_status(
            host="http://localhost:11434", model_name="llama3.1:8b"
        )
        self.assertTrue(ok)
        self.assertIn("ready", msg)

    @patch("urllib.request.urlopen")
    def test_check_ollama_status_model_missing(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.read.return_value = json.dumps(
            {"models": [{"name": "mistral:7b"}]}
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        ok, msg = check_ollama_status(
            host="http://localhost:11434", model_name="llama3.1:8b"
        )
        self.assertFalse(ok)
        self.assertIn("was not found", msg)
        self.assertIn("scripts/pull-model.sh", msg)

    @patch("urllib.request.urlopen")
    def test_check_ollama_status_unreachable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        ok, msg = check_ollama_status(
            host="http://localhost:11434", model_name="llama3.1:8b"
        )
        self.assertFalse(ok)
        self.assertIn("not reachable", msg)
        self.assertIn("docker compose up -d", msg)

    def test_retry_with_backoff_success(self) -> None:
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise ValueError("Transient error")
            return "ok"

        with patch("time.sleep", return_value=None):
            result = retry_with_backoff(flaky, max_retries=3, backoff_factor=1.0)
            self.assertEqual(result, "ok")
            self.assertEqual(calls, 2)

    def test_retry_with_backoff_exhausted(self) -> None:
        def fails() -> None:
            raise ValueError("Permanent error")

        with patch("time.sleep", return_value=None):
            with self.assertRaises(ValueError):
                retry_with_backoff(fails, max_retries=2, backoff_factor=1.0)


if __name__ == "__main__":
    unittest.main()

