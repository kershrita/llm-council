"""OpenRouter API client for making LLM requests."""

import asyncio
import logging
import httpx
from time import perf_counter
from typing import List, Dict, Any, Optional
from .config import (
    OPENROUTER_API_KEYS,
    OPENROUTER_API_URL,
    MAX_MODEL_RETRIES,
    MODEL_RETRY_BASE_DELAY_SECONDS,
    MODEL_REQUEST_TIMEOUT_SECONDS,
    MAX_FALLBACK_MODELS,
)


logger = logging.getLogger(__name__)


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


async def _query_model_once(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float,
    api_key: str,
) -> Dict[str, Any]:
    """Execute a single OpenRouter request without retries."""
    headers = {
        "Authorization": f"Bearer {api_key}",
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
    trace_id: Optional[str] = None,
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
    trace_value = trace_id or "-"

    if not OPENROUTER_API_KEYS:
        logger.error(
            "Model query skipped due to missing API keys trace_id=%s requested_model=%s",
            trace_value,
            model,
        )
        return {
            "ok": False,
            "requested_model": model,
            "model": model,
            "content": None,
            "reasoning_details": None,
            "status_code": None,
                "error": "No OpenRouter API key is set (OPENROUTER_API_KEY / OPENROUTER_API_KEYS / KEY1,KEY2,...)",
            "attempted_models": [model],
            "fallback_used": False,
            "rate_limited": False,
            "rate_limit_events": [],
                "attempted_key_slots": [],
                "api_key_slot": None,
        }

    api_keys = OPENROUTER_API_KEYS

    candidate_models = [model]
    if fallback_models:
        limited_fallbacks = fallback_models[:MAX_FALLBACK_MODELS]
        candidate_models.extend(
            fallback for fallback in limited_fallbacks if fallback not in candidate_models
        )

    logger.info(
        "Model query start trace_id=%s requested_model=%s candidate_count=%d api_key_count=%d timeout_s=%.1f max_retries=%d",
        trace_value,
        model,
        len(candidate_models),
        len(api_keys),
        timeout,
        max_retries,
    )

    attempted_models: List[str] = []
    attempted_key_slots: List[int] = []
    rate_limit_events: List[Dict[str, Any]] = []
    last_error = "Unknown error"
    last_status_code: Optional[int] = None
    stop_after_current_candidate = False
    request_started = perf_counter()

    for candidate_model in candidate_models:
        if stop_after_current_candidate:
            logger.warning(
                "Model query halted after terminal condition trace_id=%s requested_model=%s attempted_models=%s",
                trace_value,
                model,
                attempted_models,
            )
            break

        attempted_models.append(candidate_model)
        if candidate_model != model:
            logger.warning(
                "Model fallback candidate selected trace_id=%s requested_model=%s fallback_model=%s",
                trace_value,
                model,
                candidate_model,
            )

        for api_key_index, api_key in enumerate(api_keys):
            api_key_slot = api_key_index + 1
            if api_key_slot not in attempted_key_slots:
                attempted_key_slots.append(api_key_slot)

            if api_key_index > 0:
                logger.warning(
                    "Model query API key rotation trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d",
                    trace_value,
                    model,
                    candidate_model,
                    api_key_slot,
                )

            switch_to_next_key = False
            for attempt in range(max_retries + 1):
                attempt_number = attempt + 1
                logger.debug(
                    "Model query attempt trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d attempt=%d",
                    trace_value,
                    model,
                    candidate_model,
                    api_key_slot,
                    attempt_number,
                )
                try:
                    payload = await _query_model_once(candidate_model, messages, timeout, api_key)
                    elapsed_ms = int((perf_counter() - request_started) * 1000)
                    logger.info(
                        "Model query success trace_id=%s requested_model=%s used_model=%s api_key_slot=%d attempt=%d fallback_used=%s elapsed_ms=%d",
                        trace_value,
                        model,
                        candidate_model,
                        api_key_slot,
                        attempt_number,
                        candidate_model != model,
                        elapsed_ms,
                    )
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
                        "rate_limited": len(rate_limit_events) > 0,
                        "rate_limit_events": rate_limit_events.copy(),
                        "attempted_key_slots": attempted_key_slots.copy(),
                        "api_key_slot": api_key_slot,
                    }

                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code if exc.response is not None else None
                    last_status_code = status_code
                    last_error = str(exc)
                    retryable = status_code in RETRYABLE_STATUS_CODES
                    is_auth_error = status_code in {401, 403}

                    if status_code == 429:
                        rate_limit_events.append({
                            "candidate_model": candidate_model,
                            "attempt": attempt_number,
                            "status_code": status_code,
                            "api_key_slot": api_key_slot,
                        })

                        if api_key_index < len(api_keys) - 1:
                            logger.warning(
                                "Model query key rate limited trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d next_api_key_slot=%d",
                                trace_value,
                                model,
                                candidate_model,
                                api_key_slot,
                                api_key_slot + 1,
                            )
                            switch_to_next_key = True
                            break

                    if is_auth_error:
                        if api_key_index < len(api_keys) - 1:
                            logger.warning(
                                "Model query key unauthorized trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d status_code=%s next_api_key_slot=%d",
                                trace_value,
                                model,
                                candidate_model,
                                api_key_slot,
                                status_code,
                                api_key_slot + 1,
                            )
                            switch_to_next_key = True
                            break

                        # If all keys are unauthorized, don't burn additional fallback models.
                        stop_after_current_candidate = True

                    if retryable and attempt < max_retries:
                        delay = retry_base_delay * (2 ** attempt)
                        logger.warning(
                            "Model query HTTP retry trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d status_code=%s attempt=%d delay_s=%.1f",
                            trace_value,
                            model,
                            candidate_model,
                            api_key_slot,
                            status_code,
                            attempt_number,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.warning(
                        "Model query HTTP failure trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d status_code=%s attempt=%d error=%s",
                        trace_value,
                        model,
                        candidate_model,
                        api_key_slot,
                        status_code,
                        attempt_number,
                        last_error,
                    )
                    break

                except Exception as exc:  # Network and timeout failures
                    last_error = str(exc)
                    last_status_code = None
                    if attempt < max_retries:
                        delay = retry_base_delay * (2 ** attempt)
                        logger.warning(
                            "Model query network retry trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d attempt=%d delay_s=%.1f error=%s",
                            trace_value,
                            model,
                            candidate_model,
                            api_key_slot,
                            attempt_number,
                            delay,
                            last_error,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # If we exhausted retries on a timeout, don't chain into extra
                    # fallback models that are likely to time out the same way.
                    if isinstance(exc, httpx.TimeoutException):
                        stop_after_current_candidate = True

                    logger.warning(
                        "Model query network failure trace_id=%s requested_model=%s candidate_model=%s api_key_slot=%d attempt=%d timeout=%s error=%s",
                        trace_value,
                        model,
                        candidate_model,
                        api_key_slot,
                        attempt_number,
                        isinstance(exc, httpx.TimeoutException),
                        last_error,
                    )
                    break

            if switch_to_next_key:
                continue

            # For non-429 failures we preserve previous behavior and move to
            # fallback model candidates (if any) instead of cycling keys.
            break

    elapsed_ms = int((perf_counter() - request_started) * 1000)
    logger.error(
        "Model query failed trace_id=%s requested_model=%s attempted_models=%s status_code=%s elapsed_ms=%d error=%s",
        trace_value,
        model,
        attempted_models,
        last_status_code,
        elapsed_ms,
        last_error,
    )
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
        "rate_limited": len(rate_limit_events) > 0,
        "rate_limit_events": rate_limit_events,
        "attempted_key_slots": attempted_key_slots,
        "api_key_slot": None,
    }


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    timeout: float = MODEL_REQUEST_TIMEOUT_SECONDS,
    max_retries: int = MAX_MODEL_RETRIES,
    fallback_models: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping requested model identifier to structured query result.
    """
    trace_value = trace_id or "-"
    started = perf_counter()
    logger.info(
        "Parallel model query start trace_id=%s model_count=%d timeout_s=%.1f",
        trace_value,
        len(models),
        timeout,
    )

    # Create tasks for all models
    tasks = [
        query_model(
            model,
            messages,
            timeout=timeout,
            max_retries=max_retries,
            fallback_models=fallback_models,
            trace_id=trace_id,
        )
        for model in models
    ]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    success_count = sum(1 for response in responses if response.get("ok"))
    elapsed_ms = int((perf_counter() - started) * 1000)
    logger.info(
        "Parallel model query complete trace_id=%s success=%d failed=%d elapsed_ms=%d",
        trace_value,
        success_count,
        len(models) - success_count,
        elapsed_ms,
    )

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
