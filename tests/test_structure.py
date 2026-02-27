"""
Basic test to verify project structure and imports work correctly.
"""

import os
import sys


def test_project_structure():
    """Test that all required modules can be imported."""

    # Add the src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, src_path)

    try:
        # Test importing core modules
        from main import app

        print("✓ main module imported successfully")

        # Test importing config
        from config import settings

        print("✓ config module imported successfully")

        # Test importing models
        from models import ModelInfo, TagsResponse

        print("✓ models module imported successfully")

        # Test importing routers
        from routers.chat import router as chat_router
        from routers.generate import router as generate_router
        from routers.health import router as health_router
        from routers.tags import router as tags_router

        print("✓ routers imported successfully")

        # Test importing services
        from services.model_aggregator import ModelAggregator
        from services.ollama_client import OllamaClient

        print("✓ services imported successfully")

        print("\n✓ All project components imported successfully!")
        return True

    except ImportError as e:
        print(f"✗ ImportError: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_project_structure()
    sys.exit(0 if success else 1)
