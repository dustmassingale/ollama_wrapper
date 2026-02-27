"""
Chat completions endpoint router for the Ollama Wrapper Gateway.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from src.services.model_aggregator import ModelAggregator

from src.config import settings
from src.models import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


# Dependency injection for model aggregator
def get_model_aggregator():
    """Provide model aggregator instance."""
    return ModelAggregator(
        remote_url=settings.remote_ollama_url,
        local_url=settings.local_ollama_url,
        remote_prefix=settings.remote_model_prefix,
        local_prefix=settings.local_model_prefix,
    )


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: Request,
    chat_request: ChatRequest,
    model_aggregator: ModelAggregator = Depends(get_model_aggregator),
):
    """
    Chat completions using the specified model.

    Args:
        chat_request: The chat request parameters

    Returns:
        ChatResponse: The chat response and metadata

    Raises:
        HTTPException: If model not found or chat fails
    """
    try:
        # Determine if the model is local or remote
        is_remote = model_aggregator.is_model_remote(chat_request.model)

        # Remove prefix to get actual model name
        actual_model_name = model_aggregator.strip_prefix(chat_request.model)

        # Set up the proxy request
        if is_remote:
            target_url = f"{settings.remote_ollama_url}/api/chat"
        else:
            target_url = f"{settings.local_ollama_url}/api/chat"

        # Prepare the request payload
        payload = chat_request.dict(exclude_unset=True)
        payload["model"] = actual_model_name

        # Perform the proxy request
        response = await model_aggregator.proxy_request(
            target_url, "POST", payload, stream=chat_request.stream
        )

        # Return the response directly
        return response

    except Exception as e:
        logger.error(f"Chat completions failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Chat completions failed: {str(e)}"
        )
