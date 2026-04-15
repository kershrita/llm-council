"""OpenRouter API client for making LLM requests."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    MAX_MODEL_RETRIES,
    MODEL_RETRY_BASE_DELAY_SECONDS,
    MODEL_REQUEST_TIMEOUT_SECONDS,
    MAX_FALLBACK_MODELS,
)


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


async def _query_model_once(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float,
) -> Dict[str, Any]:
    """Execute a single OpenRouter request without retries."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    client_timeout = httpx.Timeout(
        timeout=timeout,
        connect=min(timeout, 10.0),
    )

    async with httpx.AsyncClient(timeout=client_timeout) as client:
        response = await client.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

    data = response.json()
    message = data["choices"][0]["message"]
    return {
        "content": message.get("content", ""),
        "reasoning_details": message.get("reasoning_details"),
    }


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = MODEL_REQUEST_TIMEOUT_SECONDS,
    max_retries: int = MAX_MODEL_RETRIES,
    retry_base_delay: float = MODEL_RETRY_BASE_DELAY_SECONDS,
    fallback_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Dict with success/error details and payload fields.
    """
    if not OPENROUTER_API_KEY:
        return {
            "ok": False,
            "requested_model": model,
            "model": model,
            "content": None,
            "reasoning_details": None,
            "status_code": None,
            "error": "OPENROUTER_API_KEY is not set",
            "attempted_models": [model],
            "fallback_used": False,
        }

    candidate_models = [model]
    if fallback_models:
        limited_fallbacks = fallback_models[:MAX_FALLBACK_MODELS]
        candidate_models.extend(
            fallback for fallback in limited_fallbacks if fallback not in candidate_models
        )

    attempted_models: List[str] = []
    last_error = "Unknown error"
    last_status_code: Optional[int] = None
    stop_after_current_candidate = False

    for candidate_model in candidate_models:
        if stop_after_current_candidate:
            break

        attempted_models.append(candidate_model)

        for attempt in range(max_retries + 1):
            try:
                payload = await _query_model_once(candidate_model, messages, timeout)
                return {
                    "ok": True,
                    "requested_model": model,
                    "model": candidate_model,
                    "content": payload.get("content", ""),
                    "reasoning_details": payload.get("reasoning_details"),
                    "status_code": 200,
                    "error": None,
                    "attempted_models": attempted_models.copy(),
                    "fallback_used": candidate_model != model,
                }

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                last_status_code = status_code
                last_error = str(exc)
                retryable = status_code in RETRYABLE_STATUS_CODES

                if retryable and attempt < max_retries:
                    delay = retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                break

            except Exception as exc:  # Network and timeout failures
                last_error = str(exc)
                last_status_code = None
                if attempt < max_retries:
                    delay = retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue

                # If we exhausted retries on a timeout, don't chain into extra
                # fallback models that are likely to time out the same way.
                if isinstance(exc, httpx.TimeoutException):
                    stop_after_current_candidate = True
                break

    print(f"Error querying model {model}: {last_error}")
    return {
        "ok": False,
        "requested_model": model,
        "model": model,
        "content": None,
        "reasoning_details": None,
        "status_code": last_status_code,
        "error": last_error,
        "attempted_models": attempted_models,
        "fallback_used": False,
    }


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    timeout: float = MODEL_REQUEST_TIMEOUT_SECONDS,
    max_retries: int = MAX_MODEL_RETRIES,
    fallback_models: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping requested model identifier to structured query result.
    """
    # Create tasks for all models
    tasks = [
        query_model(
            model,
            messages,
            timeout=timeout,
            max_retries=max_retries,
            fallback_models=fallback_models,
        )
        for model in models
    ]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
