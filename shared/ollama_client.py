"""Ollama server health checks, model verification, and retry utilities."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


def check_ollama_status(
    host: str = "http://localhost:11434",
    model_name: str = "llama3.1:8b",
    timeout: int = 5,
) -> tuple[bool, str]:
    """Verify Ollama server reachability and ensure the specified model is installed.

    Returns:
        (is_ok, message): True and confirmation message if ready, False and helpful advice otherwise.
    """
    clean_host = host.rstrip("/")
    tags_url = f"{clean_host}/api/tags"

    # Step 1: Reachability check
    try:
        req = urllib.request.Request(
            tags_url,
            headers={"User-Agent": "wiki-research-agents/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.getcode()
            if status_code != 200:
                return (
                    False,
                    f"Ollama server at {clean_host} returned unexpected HTTP {status_code}.",
                )
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as err:
        return (
            False,
            f"Ollama server is not reachable at '{clean_host}' ({err.reason}).\n"
            "Action required:\n"
            "  1. Start the container: docker compose up -d\n"
            f"  2. Or if running natively on macOS: ollama serve\n"
            f"  3. Pull the model: ./scripts/pull-model.sh {model_name}",
        )
    except Exception as err:
        return False, f"Failed connecting to Ollama at '{clean_host}': {err}"

    # Step 2: Model availability check
    models = [m.get("name", "") for m in data.get("models", [])]

    model_base = model_name.split(":")[0]
    matched = False
    for m in models:
        if m == model_name or m.startswith(f"{model_name}:") or m.split(":")[0] == model_base:
            matched = True
            break

    if not matched:
        available_str = ", ".join(models) if models else "none"
        return (
            False,
            f"Model '{model_name}' was not found in Ollama.\n"
            f"Installed models: [{available_str}]\n"
            f"Action required:\n"
            f"  Run: ./scripts/pull-model.sh {model_name}\n"
            f"  Or in Docker: docker compose exec ollama ollama pull {model_name}",
        )

    return True, f"Ollama is reachable at {clean_host} and model '{model_name}' is ready."


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute a function with exponential backoff on transient errors."""
    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except exceptions as exc:
            last_exception = exc
            if attempt == max_retries:
                logger.error(
                    "Call failed after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise
            logger.warning(
                "Attempt %d/%d failed with error: %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= backoff_factor

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop exited unexpectedly without result or exception")

