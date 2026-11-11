"""Configuration loading and validation for Ollama and agent pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set.

    Avoids strict dependency on python-dotenv while fully supporting .env files.
    """
    if dotenv_path is None:
        dotenv_path = Path(".env")

    if not dotenv_path.is_file():
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val
    except OSError:
        pass


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment or .env."""

    ollama_host: str = "http://localhost:11434"
    model_name: str = "llama3.1:8b"
    temperature: float = 0.2
    request_timeout: int = 120
    max_tool_calls: int = 3
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Settings:
        """Create Settings instance populated from environment and .env."""
        load_dotenv(env_path)
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            model_name=os.getenv("MODEL_NAME", "llama3.1:8b"),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "3")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_backoff_factor=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
        )


_SETTINGS_INSTANCE: Settings | None = None


def get_config(reload: bool = False, env_path: Path | None = None) -> Settings:
    """Return the cached singleton Settings instance."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None or reload:
        _SETTINGS_INSTANCE = Settings.from_env(env_path)
    return _SETTINGS_INSTANCE

