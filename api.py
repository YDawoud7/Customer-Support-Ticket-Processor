"""FastAPI wrapper for the ticket processing graph."""

import json
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from src.checkpointer import get_checkpointer  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.models import PRESET_CONFIGS  # noqa: E402

graph_app = None
checkpointer = None

INITIAL_STATE = {
    "messages": [],
    "ticket_text": "",
    "category": "",
    "confidence": 0.0,
    "reasoning": "",
    "response": "",
    "retrieved_docs": [],
    "quality_approved": False,
    "quality_feedback": "",
}


class TicketRequest(BaseModel):
    ticket_text: str
    model_config_name: str | None = None


class ResumeRequest(BaseModel):
    value: str


class InterruptPayload(BaseModel):
    type: str
    message: str
    options: list[str]
    current_category: str | None = None
    reasoning: str | None = None
    draft_response: str | None = None
    quality_feedback: str | None = None


class TicketResponse(BaseModel):
    thread_id: str
    status: str
    category: str
    confidence: float
    reasoning: str
    response: str
    retrieved_docs: list[str]
    quality_approved: bool
    quality_feedback: str
    interrupt: InterruptPayload | None = None


def _check_interrupt(graph_state):
    """Check graph state for pending interrupts and return payload if found."""
    if not graph_state.tasks:
        return None
    for task in graph_state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            info = task.interrupts[0].value
            if "current_category" in info:
                return InterruptPayload(
                    type="confidence",
                    message=info["message"],
                    options=info["options"],
                    current_category=info["current_category"],
                    reasoning=info["reasoning"],
                )
            else:
                return InterruptPayload(
                    type="quality",
                    message=info["message"],
                    options=info["options"],
                    draft_response=info.get("draft_response"),
                    quality_feedback=info.get("quality_feedback"),
                )
    return None


def _state_to_response(thread_id, result, interrupt=None):
    """Convert graph result dict to TicketResponse."""
    return TicketResponse(
        thread_id=thread_id,
        status="interrupted" if interrupt else "completed",
        category=result.get("category", ""),
        confidence=result.get("confidence", 0.0),
        reasoning=result.get("reasoning", ""),
        response=result.get("response", ""),
        retrieved_docs=result.get("retrieved_docs", []),
        quality_approved=result.get("quality_approved", False),
        quality_feedback=result.get("quality_feedback", ""),
        interrupt=interrupt,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_app, checkpointer
    checkpointer = get_checkpointer()
    graph_app = build_graph(checkpointer=checkpointer)
    yield


app = FastAPI(title="Customer Support Ticket Processor", lifespan=lifespan)


@app.get("/health")
def health():
    redis_connected = False
    try:
        from langgraph.checkpoint.redis import RedisSaver

        if isinstance(checkpointer, RedisSaver):
            redis_connected = True
    except ImportError:
        pass
    return {"status": "ok", "redis": redis_connected}


@app.post("/tickets", response_model=TicketResponse, status_code=200)
def submit_ticket(request: TicketRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    model_config = None
    if request.model_config_name:
        model_config = PRESET_CONFIGS.get(request.model_config_name)
        if not model_config:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown config: {request.model_config_name}. Options: {list(PRESET_CONFIGS.keys())}",
            )

    if model_config:
        local_graph = build_graph(model_config=model_config, checkpointer=checkpointer)
    else:
        local_graph = graph_app

    state = {**INITIAL_STATE, "ticket_text": request.ticket_text}
    result = local_graph.invoke(state, config=config)

    graph_state = local_graph.get_state(config)
    interrupt = _check_interrupt(graph_state)

    resp = _state_to_response(thread_id, result, interrupt)
    return resp


@app.post("/tickets/{thread_id}/resume", response_model=TicketResponse)
def resume_ticket(thread_id: str, request: ResumeRequest):
    config = {"configurable": {"thread_id": thread_id}}

    try:
        graph_state = graph_app.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="Thread not found")

    if not graph_state.values:
        raise HTTPException(status_code=404, detail="Thread not found")

    result = graph_app.invoke(Command(resume=request.value), config=config)

    graph_state = graph_app.get_state(config)
    interrupt = _check_interrupt(graph_state)

    return _state_to_response(thread_id, result, interrupt)


@app.get("/tickets/{thread_id}", response_model=TicketResponse)
def get_ticket(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    try:
        graph_state = graph_app.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="Thread not found")

    if not graph_state.values:
        raise HTTPException(status_code=404, detail="Thread not found")

    interrupt = _check_interrupt(graph_state)
    return _state_to_response(thread_id, graph_state.values, interrupt)


@app.post("/tickets/stream")
def stream_ticket(request: TicketRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state = {**INITIAL_STATE, "ticket_text": request.ticket_text}

    def event_generator():
        yield {
            "event": "start",
            "data": json.dumps({"thread_id": thread_id}),
        }

        for node_name, update in graph_app.stream(state, config=config, stream_mode="updates"):
            safe_update = {
                k: v for k, v in update.items() if k != "messages"
            }
            yield {
                "event": "node_update",
                "data": json.dumps({"node": node_name, "update": safe_update}),
            }

        graph_state = graph_app.get_state(config)
        interrupt = _check_interrupt(graph_state)

        if interrupt:
            yield {
                "event": "interrupt",
                "data": json.dumps({
                    "thread_id": thread_id,
                    "interrupt": interrupt.model_dump(),
                }),
            }
        else:
            result = graph_state.values
            yield {
                "event": "complete",
                "data": json.dumps({
                    "thread_id": thread_id,
                    "category": result.get("category", ""),
                    "confidence": result.get("confidence", 0.0),
                    "response": result.get("response", ""),
                    "quality_approved": result.get("quality_approved", False),
                }),
            }

    return EventSourceResponse(event_generator())
