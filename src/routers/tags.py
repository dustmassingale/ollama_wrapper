"""
Tags endpoint router for the Ollama Wrapper Gateway.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.services.model_aggregator import ModelAggregator

from src.models import TagsResponse

router = APIRouter()


# Dependency injection for model aggregator
def get_model_aggregator():
    """Provide model aggregator instance."""
    return ModelAggregator()


@router.get("/", response_model=TagsResponse)
async def list_models(aggregator: ModelAggregator = Depends(get_model_aggregator)):
    """
    Get all available models from both local and remote Ollama instances.

    Returns:
        TagsResponse: List of all available models with appropriate prefixes.
    """
    try:
        models = await aggregator.get_all_models()
        return TagsResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")
