"""
/api/generate endpoint router for the Ollama Wrapper Gateway.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.config import settings
from src.models import GenerateRequest, GenerateResponse
from src.services.model_aggregator import ModelAggregator

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


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: Request,
    generate_request: GenerateRequest,
    model_aggregator: ModelAggregator = Depends(get_model_aggregator),
):
    """
    Generate text using the specified model.

    Args:
        request: FastAPI request object
        generate_request: Request parameters
        model_aggregator: Model aggregator instance

    Returns:
        GenerateResponse: Generated text response

    Raises:
        HTTPException: If model not found or generation fails
    """
    try:
        # Validate that model has the prefix and gets actual model name
        is_remote = model_aggregator.is_model_remote(generate_request.model)
        actual_model_name = model_aggregator.strip_prefix(generate_request.model)

        # Check if model name looks valid
        if not actual_model_name:
            raise HTTPException(status_code=400, detail="Invalid model name")

        # Get the raw request body
        raw_body = await request.body()

        # Forward to appropriate backend
        response = await model_aggregator.generate_text(
            generate_request, generate_request.stream
        )

        return response
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
