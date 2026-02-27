"""
Service to aggregate models from local and remote Ollama instances.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from src.config import settings
from src.models import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
)
from src.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ModelAggregator:
    """Aggregates models from local and remote Ollama instances."""

    def __init__(
        self,
        remote_url: str = None,
        local_url: str = None,
        remote_prefix: str = None,
        local_prefix: str = None,
    ):
        """
        Initialize the model aggregator.

        Args:
            remote_url: URL of remote Ollama instance
            local_url: URL of local Ollama instance
            remote_prefix: Prefix for remote models
            local_prefix: Prefix for local models
        """
        self.remote_url = remote_url or settings.remote_ollama_url
        self.local_url = local_url or settings.local_ollama_url
        self.remote_prefix = remote_prefix or settings.remote_model_prefix
        self.local_prefix = local_prefix or settings.local_model_prefix

        # Initialize clients
        self.remote_client = OllamaClient(self.remote_url)
        self.local_client = OllamaClient(self.local_url)

        # Cache for models
        self._cached_models = []
        self._last_cache_time = 0
        self._cache_timeout = settings.cache_timeout

    async def close(self):
        """Clean up resources."""
        await self.remote_client.close()
        await self.local_client.close()

    async def get_remote_models(self) -> List[ModelInfo]:
        """
        Get models from the remote Ollama instance.

        Returns:
            List of ModelInfo objects from remote instance
        """
        try:
            response = await self.remote_client.get("/api/tags")
            data = response.json()

            models = []
            for model in data.get("models", []):
                # Add prefix to remote models
                prefixed_name = f"{self.remote_prefix}{model['name']}"
                model_info = ModelInfo(
                    name=prefixed_name,
                    modified_at=model.get("modified_at"),
                    size=model.get("size"),
                )
                models.append(model_info)

            return models
        except Exception as e:
            logger.error(f"Failed to get remote models: {str(e)}")
            return []

    async def get_local_models(self) -> List[ModelInfo]:
        """
        Get models from the local Ollama instance.

        Returns:
            List of ModelInfo objects from local instance
        """
        try:
            response = await self.local_client.get("/api/tags")
            data = response.json()

            models = []
            for model in data.get("models", []):
                # Add prefix to local models
                prefixed_name = f"{self.local_prefix}{model['name']}"
                model_info = ModelInfo(
                    name=prefixed_name,
                    modified_at=model.get("modified_at"),
                    size=model.get("size"),
                )
                models.append(model_info)

            return models
        except Exception as e:
            logger.error(f"Failed to get local models: {str(e)}")
            return []

    async def get_all_models(self) -> List[ModelInfo]:
        """
        Get all models from both local and remote instances.

        Returns:
            List of all available models with appropriate prefixes
        """
        # Check if cached models are still valid
        import time

        current_time = time.time()
        if current_time - self._last_cache_time < self._cache_timeout:
            return self._cached_models

        # Get models from both sources concurrently
        try:
            remote_models_task = self.get_remote_models()
            local_models_task = self.get_local_models()

            remote_models, local_models = await asyncio.gather(
                remote_models_task, local_models_task, return_exceptions=True
            )

            # Handle potential exceptions
            if isinstance(remote_models, Exception):
                logger.error(f"Error fetching remote models: {remote_models}")
                remote_models = []

            if isinstance(local_models, Exception):
                logger.error(f"Error fetching local models: {local_models}")
                local_models = []

            # Combine models
            all_models = [*remote_models, *local_models]

            # Cache the results
            self._cached_models = all_models
            self._last_cache_time = current_time

            return all_models
        except Exception as e:
            logger.error(f"Failed to get all models: {str(e)}")
            return []

    def is_model_remote(self, model_name: str) -> bool:
        """
        Determine if a model belongs to the remote instance.

        Args:
            model_name: Full model name (including prefix)

        Returns:
            True if model is from remote instance, False otherwise
        """
        return model_name.startswith(self.remote_prefix)

    def strip_prefix(self, model_name: str) -> str:
        """
        Strip the prefix from a model name to get the actual model name.

        Args:
            model_name: Full model name (including prefix)

        Returns:
            Model name without prefix
        """
        if model_name.startswith(self.remote_prefix):
            return model_name[len(self.remote_prefix) :]
        elif model_name.startswith(self.local_prefix):
            return model_name[len(self.local_prefix) :]
        return model_name

    async def check_remote_connection(self) -> bool:
        """
        Check connection to remote Ollama instance.

        Returns:
            True if connection is successful, False otherwise
        """
        return await self.remote_client.ping()

    async def check_local_connection(self) -> bool:
        """
        Check connection to local Ollama instance.

        Returns:
            True if connection is successful, False otherwise
        """
        return await self.local_client.ping()

    async def generate_text(
        self, request: GenerateRequest, stream: bool = False
    ) -> Union[GenerateResponse, Any]:
        """
        Generate text using the specified model by routing to correct backend.

        Args:
            request: Generation request
            stream: Whether to enable streaming

        Returns:
            GenerateResponse or streaming response
        """
        # Determine which backend to use based on prefix
        is_remote = self.is_model_remote(request.model)
        actual_model_name = self.strip_prefix(request.model)

        # Set the correct backend client
        if is_remote:
            client = self.remote_client
            target_url = f"{self.remote_url}/api/generate"
        else:
            client = self.local_client
            target_url = f"{self.local_url}/api/generate"

        # Prepare the payload without the prefix
        payload = request.dict(exclude_unset=True)
        payload["model"] = actual_model_name

        # Perform the proxy request
        try:
            if stream:
                # Streaming is handled via the ollama client, but we need to make sure
                # we're setting the right stream flag in client
                response = await client.post(
                    "/api/generate", json_data=payload, stream=stream
                )
                return response
            else:
                response = await client.post("/api/generate", json_data=payload)
                response_data = response.json()
                return GenerateResponse(**response_data)
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise

    async def chat_completion(
        self, request: ChatRequest, stream: bool = False
    ) -> Union[ChatResponse, Any]:
        """
        Chat completion using the specified model by routing to correct backend.

        Args:
            request: Chat request
            stream: Whether to enable streaming

        Returns:
            ChatResponse or streaming response
        """
        # Determine which backend to use based on prefix
        is_remote = self.is_model_remote(request.model)
        actual_model_name = self.strip_prefix(request.model)

        # Set the correct backend URL and client
        if is_remote:
            client = self.remote_client
            target_url = f"{self.remote_url}/api/chat"
        else:
            client = self.local_client
            target_url = f"{self.local_url}/api/chat"

        # Prepare the payload without the prefix
        payload = request.dict(exclude_unset=True)
        payload["model"] = actual_model_name

        # Perform the proxy request
        try:
            if stream:
                # Streaming is handled appropriately
                response = await client.post(
                    "/api/chat", json_data=payload, stream=stream
                )
                return response
            else:
                response = await client.post("/api/chat", json_data=payload)
                response_data = response.json()
                return ChatResponse(**response_data)
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise
