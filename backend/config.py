"""Configuration for the LLM Council."""

import os
import re
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Logging level for backend modules (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def _append_unique_key(keys: List[str], raw_value: str | None) -> None:
    """Add a key if present and not already included."""
    if not raw_value:
        return

    value = raw_value.strip()
    if value and value not in keys:
        keys.append(value)


def _collect_openrouter_api_keys() -> List[str]:
    """Collect OpenRouter API keys from supported environment variables."""
    keys: List[str] = []

    # Primary single-key variable.
    _append_unique_key(keys, os.getenv("OPENROUTER_API_KEY"))

    # Optional comma-separated list.
    multi_value = os.getenv("OPENROUTER_API_KEYS")
    if multi_value:
        for chunk in multi_value.split(","):
            _append_unique_key(keys, chunk)

    numbered_key_values: List[tuple[int, str]] = []

    # Support OPENROUTER_API_KEY1, OPENROUTER_API_KEY2, ...
    # and KEY1, KEY2, ... for convenience.
    for env_name, env_value in os.environ.items():
        openrouter_match = re.fullmatch(r"OPENROUTER_API_KEY(\d+)", env_name, flags=re.IGNORECASE)
        if openrouter_match:
            numbered_key_values.append((int(openrouter_match.group(1)), env_value))
            continue

        generic_match = re.fullmatch(r"KEY(\d+)", env_name, flags=re.IGNORECASE)
        if generic_match:
            numbered_key_values.append((10_000 + int(generic_match.group(1)), env_value))

    for _, value in sorted(numbered_key_values, key=lambda item: item[0]):
        _append_unique_key(keys, value)

    return keys


# OpenRouter API keys (primary + optional fallbacks)
OPENROUTER_API_KEYS = _collect_openrouter_api_keys()

# Backward-compatible alias for older code paths.
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else None

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "openai/gpt-oss-20b:free",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "openai/gpt-oss-120b:free"

# Model used for short conversation title generation
TITLE_MODEL = "openai/gpt-oss-20b:free"

# Backup models used if a requested model is rate-limited or unavailable.
FALLBACK_MODELS = [
    "openrouter/free",
    "qwen/qwen3-coder:free",
    "z-ai/glm-4.5-air:free",
]

# Retry settings for transient OpenRouter failures (429/5xx/timeouts).
MAX_MODEL_RETRIES = int(os.getenv("MAX_MODEL_RETRIES", "0"))
MODEL_RETRY_BASE_DELAY_SECONDS = float(os.getenv("MODEL_RETRY_BASE_DELAY_SECONDS", "1.0"))

# Runtime limits to prevent long silent waits.
MODEL_REQUEST_TIMEOUT_SECONDS = float(os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", "30"))
TITLE_REQUEST_TIMEOUT_SECONDS = float(os.getenv("TITLE_REQUEST_TIMEOUT_SECONDS", "20"))
MAX_FALLBACK_MODELS = int(os.getenv("MAX_FALLBACK_MODELS", "1"))

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
