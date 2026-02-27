"""
Configuration management for the Ollama Wrapper Gateway.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Remote Ollama Server Configuration
    remote_ollama_url: str = Field(
        default="http://192.168.1.155:11434",
        description="URL of the remote Ollama server",
    )

    # Local Ollama Server Configuration
    local_ollama_url: str = Field(
        default="http://localhost:11434", description="URL of the local Ollama server"
    )

    # Gateway Configuration
    gateway_host: str = Field(default="0.0.0.0", description="Gateway hostname")
    gateway_port: int = Field(default=8000, description="Gateway port")

    # Performance Settings
    cache_timeout: int = Field(default=300, description="Cache timeout in seconds")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")

    # Model prefix configuration
    remote_model_prefix: str = Field(
        default="155-", description="Prefix for remote models"
    )
    local_model_prefix: str = Field(
        default="LOC-", description="Prefix for local models"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()
