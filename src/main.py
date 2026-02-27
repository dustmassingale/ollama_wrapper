"""
Main FastAPI application entry point for the Ollama Wrapper Gateway.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routers import chat, generate, health, tags
from src.services.model_aggregator import ModelAggregator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifecycle management for the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup logic
    logger.info("Starting Ollama Wrapper Gateway...")

    # Initialize aggregator for global use
    app.state.aggregator = ModelAggregator()

    yield

    # Cleanup logic
    logger.info("Shutting down Ollama Wrapper Gateway...")
    if hasattr(app.state, "aggregator"):
        await app.state.aggregator.close()


# Create the FastAPI app with lifespan
app = FastAPI(
    title="Ollama Wrapper Gateway",
    description="A gateway proxy service that aggregates models from local and remote Ollama instances.",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tags.router, prefix="/tags", tags=["tags"])
app.include_router(generate.router, prefix="/api", tags=["generate"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(health.router, prefix="/health", tags=["health"])


@app.get("/")
async def root():
    """Root endpoint providing basic information about the gateway."""
    return {
        "message": "Ollama Wrapper Gateway",
        "version": "0.1.0",
        "documentation": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        reload=True,
    )
