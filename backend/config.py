"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Logging level for backend modules (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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
