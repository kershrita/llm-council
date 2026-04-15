"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings
from .config import COUNCIL_MODELS

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
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message
    is_first_message = len(conversation["messages"]) == 0

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content
    )

    # Add assistant message with all stages
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return the complete response with metadata
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
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

            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(generate_conversation_title(request.content))

            # Stage 1: Collect responses
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results, stage1_failures = await stage1_collect_responses(request.content)
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
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results, 'metadata': metadata})}\n\n"

            if not stage1_results:
                stage3_result = {
                    "model": "error",
                    "response": "All models failed to respond. Please try again.",
                }
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
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model, stage2_failures = await stage2_collect_rankings(request.content, stage1_results)
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
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': metadata})}\n\n"

            # Stage 3: Synthesize final answer
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result, stage3_failure = await stage3_synthesize_final(request.content, stage1_results, stage2_results)
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
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result, 'metadata': metadata})}\n\n"

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )
            assistant_saved = True

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            error_message = str(e)
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
                    pass

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
