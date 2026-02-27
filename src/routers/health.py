"""
/health endpoint router for the Ollama Wrapper Gateway.
"""

import logging

from fastapi import APIRouter
from src.services.model_aggregator import ModelAggregator

from src.config import settings
from src.models import HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify gateway and backend connectivity.

    Returns:
        HealthResponse: Status of gateway and connected Ollama instances
    """
    try:
        # Check if we can connect to both Ollama instances
        aggregator = ModelAggregator(
            remote_url=settings.remote_ollama_url,
            local_url=settings.local_ollama_url,
            remote_prefix=settings.remote_model_prefix,
            local_prefix=settings.local_model_prefix,
        )

        # Check remote Ollama connection
        remote_status = "ok" if await aggregator.check_remote_connection() else "error"

        # Check local Ollama connection
        local_status = "ok" if await aggregator.check_local_connection() else "error"

        # Overall status
        status = (
            "healthy"
            if (remote_status == "ok" and local_status == "ok")
            else "unhealthy"
        )

        return HealthResponse(
            status=status,
            gateway="ok",
            remote_ollama=remote_status,
            local_ollama=local_status,
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            gateway="error",
            remote_ollama="error",
            local_ollama="error",
        )
