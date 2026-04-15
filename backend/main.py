"""FastAPI backend for LLM Council."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio
from time import perf_counter

from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
from .config import COUNCIL_MODELS, LOG_LEVEL


def configure_logging() -> None:
    """Configure root logging for backend modules."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root_logger.setLevel(level)


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    conversations = storage.list_conversations()
    logger.info("List conversations count=%d", len(conversations))
    return conversations


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    logger.info("Conversation created conversation_id=%s", conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        logger.warning("Conversation not found conversation_id=%s", conversation_id)
        raise HTTPException(status_code=404, detail="Conversation not found")
    logger.info(
        "Conversation loaded conversation_id=%s message_count=%d",
        conversation_id,
        len(conversation.get("messages", [])),
    )
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    trace_id = uuid.uuid4().hex[:8]
    request_started = perf_counter()
    logger.info(
        "Message request start trace_id=%s conversation_id=%s stream=false content_chars=%d",
        trace_id,
        conversation_id,
        len(request.content),
    )

    try:
        # Check if conversation exists
        conversation = storage.get_conversation(conversation_id)
        if conversation is None:
            logger.warning(
                "Message request conversation missing trace_id=%s conversation_id=%s",
                trace_id,
                conversation_id,
            )
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if this is the first message
        is_first_message = len(conversation["messages"]) == 0

        # Add user message
        storage.add_user_message(conversation_id, request.content)

        # If this is the first message, generate a title
        if is_first_message:
            title = await generate_conversation_title(request.content, trace_id=trace_id)
            storage.update_conversation_title(conversation_id, title)

        # Run the 3-stage council process
        stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
            request.content,
            trace_id=trace_id,
        )

        # Add assistant message with all stages
        storage.add_assistant_message(
            conversation_id,
            stage1_results,
            stage2_results,
            stage3_result
        )

        elapsed_ms = int((perf_counter() - request_started) * 1000)
        logger.info(
            "Message request complete trace_id=%s conversation_id=%s stage1=%d stage2=%d stage3_model=%s elapsed_ms=%d",
            trace_id,
            conversation_id,
            len(stage1_results),
            len(stage2_results),
            stage3_result.get("model"),
            elapsed_ms,
        )

        # Return the complete response with metadata
        return {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "stage3": stage3_result,
            "metadata": metadata
        }

    except Exception:
        logger.exception(
            "Message request failed trace_id=%s conversation_id=%s",
            trace_id,
            conversation_id,
        )
        raise


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    trace_id = uuid.uuid4().hex[:8]
    request_started = perf_counter()
    logger.info(
        "Message stream request start trace_id=%s conversation_id=%s content_chars=%d",
        trace_id,
        conversation_id,
        len(request.content),
    )

    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        logger.warning(
            "Message stream conversation missing trace_id=%s conversation_id=%s",
            trace_id,
            conversation_id,
        )
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        metadata = {
            "requested_models": COUNCIL_MODELS,
            "label_to_model": {},
            "aggregate_rankings": [],
            "failures": {
                "stage1": [],
                "stage2": [],
                "stage3": [],
            },
            "fallbacks": {
                "stage1": [],
                "stage2": [],
                "stage3": [],
            },
        }
        stage1_results: List[Dict[str, Any]] = []
        stage2_results: List[Dict[str, Any]] = []
        stage3_result: Dict[str, Any] = {
            "model": "error",
            "response": "The council request did not complete.",
        }
        assistant_saved = False

        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content)
            logger.info(
                "Stream user message saved trace_id=%s conversation_id=%s",
                trace_id,
                conversation_id,
            )

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                logger.info("Stream title task scheduled trace_id=%s", trace_id)
                title_task = asyncio.create_task(
                    generate_conversation_title(request.content, trace_id=trace_id)
                )

            # Stage 1: Collect responses
            stage1_started = perf_counter()
            logger.info("Stream stage1 start trace_id=%s", trace_id)
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results, stage1_failures = await stage1_collect_responses(
                request.content,
                trace_id=trace_id,
            )
            metadata["failures"]["stage1"] = stage1_failures
            metadata["fallbacks"]["stage1"] = [
                {
                    "requested_model": result.get("requested_model") or result.get("model"),
                    "used_model": result.get("actual_model") or result.get("model"),
                }
                for result in stage1_results
                if (result.get("requested_model") or result.get("model"))
                and (result.get("actual_model") or result.get("model"))
                and (result.get("requested_model") or result.get("model")) != (result.get("actual_model") or result.get("model"))
            ]
            stage1_elapsed_ms = int((perf_counter() - stage1_started) * 1000)
            logger.info(
                "Stream stage1 complete trace_id=%s success=%d failures=%d elapsed_ms=%d",
                trace_id,
                len(stage1_results),
                len(stage1_failures),
                stage1_elapsed_ms,
            )
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results, 'metadata': metadata})}\n\n"

            if not stage1_results:
                stage3_result = {
                    "model": "error",
                    "response": "All models failed to respond. Please try again.",
                }
                logger.warning(
                    "Stream ended early after stage1 trace_id=%s conversation_id=%s",
                    trace_id,
                    conversation_id,
                )
                yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result, 'metadata': metadata})}\n\n"
                storage.add_assistant_message(
                    conversation_id,
                    stage1_results,
                    [],
                    stage3_result,
                )
                assistant_saved = True
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                return

            # Stage 2: Collect rankings
            stage2_started = perf_counter()
            logger.info("Stream stage2 start trace_id=%s", trace_id)
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model, stage2_failures = await stage2_collect_rankings(
                request.content,
                stage1_results,
                trace_id=trace_id,
            )
            metadata["label_to_model"] = label_to_model
            metadata["aggregate_rankings"] = calculate_aggregate_rankings(stage2_results, label_to_model)
            metadata["failures"]["stage2"] = stage2_failures
            metadata["fallbacks"]["stage2"] = [
                {
                    "requested_model": result.get("requested_model") or result.get("model"),
                    "used_model": result.get("actual_model") or result.get("model"),
                }
                for result in stage2_results
                if (result.get("requested_model") or result.get("model"))
                and (result.get("actual_model") or result.get("model"))
                and (result.get("requested_model") or result.get("model")) != (result.get("actual_model") or result.get("model"))
            ]
            stage2_elapsed_ms = int((perf_counter() - stage2_started) * 1000)
            logger.info(
                "Stream stage2 complete trace_id=%s success=%d failures=%d elapsed_ms=%d",
                trace_id,
                len(stage2_results),
                len(stage2_failures),
                stage2_elapsed_ms,
            )
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': metadata})}\n\n"

            # Stage 3: Synthesize final answer
            stage3_started = perf_counter()
            logger.info("Stream stage3 start trace_id=%s", trace_id)
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result, stage3_failure = await stage3_synthesize_final(
                request.content,
                stage1_results,
                stage2_results,
                trace_id=trace_id,
            )
            if stage3_failure is not None:
                metadata["failures"]["stage3"] = [stage3_failure]
            metadata["fallbacks"]["stage3"] = []
            requested_stage3_model = stage3_result.get("requested_model") or stage3_result.get("model")
            actual_stage3_model = stage3_result.get("actual_model") or stage3_result.get("model")
            if requested_stage3_model and actual_stage3_model and requested_stage3_model != actual_stage3_model:
                metadata["fallbacks"]["stage3"] = [{
                    "requested_model": requested_stage3_model,
                    "used_model": actual_stage3_model,
                }]
            stage3_elapsed_ms = int((perf_counter() - stage3_started) * 1000)
            logger.info(
                "Stream stage3 complete trace_id=%s failures=%d elapsed_ms=%d",
                trace_id,
                len(metadata["failures"]["stage3"]),
                stage3_elapsed_ms,
            )
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result, 'metadata': metadata})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                logger.info(
                    "Stream title updated trace_id=%s conversation_id=%s title=%s",
                    trace_id,
                    conversation_id,
                    title,
                )
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )
            assistant_saved = True

            total_elapsed_ms = int((perf_counter() - request_started) * 1000)
            logger.info(
                "Message stream request complete trace_id=%s conversation_id=%s elapsed_ms=%d",
                trace_id,
                conversation_id,
                total_elapsed_ms,
            )

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            error_message = str(e)
            logger.exception(
                "Message stream request failed trace_id=%s conversation_id=%s",
                trace_id,
                conversation_id,
            )
            metadata["failures"]["stage3"] = [{
                "model": "stream",
                "status_code": None,
                "error": error_message,
                "attempted_models": [],
            }]
            stage3_result = {
                "model": "error",
                "requested_model": "error",
                "actual_model": None,
                "response": f"Error: {error_message}",
            }

            if not assistant_saved:
                try:
                    storage.add_assistant_message(
                        conversation_id,
                        stage1_results,
                        stage2_results,
                        stage3_result,
                    )
                    assistant_saved = True
                except Exception:
                    logger.exception(
                        "Failed to persist stream error assistant message trace_id=%s conversation_id=%s",
                        trace_id,
                        conversation_id,
                    )

            yield f"data: {json.dumps({'type': 'error', 'message': error_message})}\n\n"
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result, 'metadata': metadata})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
