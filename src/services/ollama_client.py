"""
HTTP client for communicating with Ollama instances.
"""
import asyncio
import httpx
from typing import Dict, Any, Optional, Union
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    """HTTP client for interacting with Ollama instances."""

    def __init__(self, base_url: str):
        """
        Initialize Ollama client.

        Args:
            base_url: Base URL of the Ollama instance (e.g., http://localhost:11434)
        """
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=settings.request_timeout,
            follow_redirects=True
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get(self, endpoint: str, params: Optional[Dict] = None) -> httpx.Response:
        """
        Make a GET request to the Ollama instance.

        Args:
            endpoint: API endpoint (e.g., '/api/tags')
            params: Query parameters

        Returns:
            httpx.Response: The HTTP response
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            logger.error(f"GET request failed to {url}: {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"GET request to {url} failed with status {e.response.status_code}: {e.response.text}")
            raise

    async def post(self, endpoint: str, json_data: Optional[Dict] = None,
                  data: Optional[Union[str, bytes]] = None, stream: bool = False) -> httpx.Response:
        """
        Make a POST request to the Ollama instance.

        Args:
            endpoint: API endpoint (e.g., '/api/generate')
            json_data: JSON payload
            data: Raw data payload
            stream: Whether to stream the response

        Returns:
            httpx.Response: The HTTP response
        """
        url = f"{self.base_url}{endpoint}"
        try:
            if stream:
                response = await self.client.post(url, json=json_data, content=data, stream=stream)
            else:
                response = await self.client.post(url, json=json_data, content=data)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            logger.error(f"POST request failed to {url}: {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"POST request to {url} failed with status {e.response.status_code}: {e.response.text}")
            raise

    async def ping(self) -> bool:
        """
        Ping the Ollama instance to check if it's responsive.

        Returns:
            bool: True if the instance is responsive, False otherwise
        """
        try:
            response = await self.get("/api/tags", params={"limit": 1})
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ping to {self.base_url} failed: {str(e)}")
            return False
</parameter>
