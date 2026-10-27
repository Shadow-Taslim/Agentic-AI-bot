"""
FastAPI backend + static frontend server for the Agentic AI Chatbot.

Run with:
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000 in your browser.

Endpoints:
    POST /api/chat          {"message": "...", "thread_id": "..."} -> {"reply": "..."}
    GET  /api/history       ?thread_id=... -> {"messages": [...]}
    GET  /api/health        -> {"status": "ok", "llm_configured": true, ...}
    GET  /                  -> frontend/index.html (chat web UI)
"""

import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agentic_chatbot_backend as backend

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = FastAPI(
    title="Agentic AI Chatbot API",
    description="LangGraph + Gemini/OpenAI chatbot with Supabase persistence",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation thread. Omit to start a new conversation.",
    )


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: List[HistoryMessage]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_configured": backend.llm is not None,
        "checkpointer": backend.checkpointer_type,
        "supabase_rest": backend.supabase_client is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if backend.llm is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM configured. Set GEMINI_API_KEY or OPENAI_API_KEY.",
        )
    thread_id = request.thread_id or f"web-{uuid.uuid4().hex[:8]}"
    try:
        reply = backend.invoke_chatbot(request.message, thread_id=thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ChatResponse(reply=reply, thread_id=thread_id)


@app.get("/api/history", response_model=HistoryResponse)
def history(thread_id: str):
    messages = backend.get_chat_history(thread_id=thread_id)
    result = []
    for msg in messages:
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        result.append(
            HistoryMessage(role=role, content=backend._content_to_text(msg.content))
        )
    return HistoryResponse(thread_id=thread_id, messages=result)


# ---------------------------------------------------------------------------
# Static frontend (mounted LAST so /api/* routes keep priority)
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
