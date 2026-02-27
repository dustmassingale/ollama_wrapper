"""
Pydantic models for request/response structures in the Ollama Wrapper Gateway.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Model information structure matching Ollama API."""

    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None
    digest: Optional[str] = None


class TagsResponse(BaseModel):
    """Response structure for /tags endpoint."""

    models: List[ModelInfo]


class GenerateRequest(BaseModel):
    """Request structure for /api/generate endpoint."""

    model: str
    prompt: str
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class GenerateResponse(BaseModel):
    """Response structure for /api/generate endpoint."""

    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ChatMessage(BaseModel):
    """Chat message structure."""

    role: str
    content: str
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Request structure for /api/chat/completions endpoint."""

    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response structure for /api/chat/completions endpoint."""

    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class HealthResponse(BaseModel):
    """Response structure for /health endpoint."""

    status: str
    gateway: str
    remote_ollama: Optional[str] = None
    local_ollama: Optional[str] = None
