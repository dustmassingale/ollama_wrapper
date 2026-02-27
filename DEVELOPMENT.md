# Development Guide - Ollama Wrapper Gateway

This guide provides instructions for setting up your development environment and contributing to the Ollama Wrapper Gateway project.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Architecture](#project-architecture)
3. [Code Structure](#code-structure)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Debugging](#debugging)
7. [Code Style & Standards](#code-style--standards)
8. [Common Development Tasks](#common-development-tasks)
9. [Troubleshooting](#troubleshooting)

## Development Environment Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- Docker & Docker Compose (optional but recommended)
- A code editor (VS Code, PyCharm, etc.)

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ollama_wrapper
   ```

2. **Create a virtual environment**
   ```bash
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-asyncio pytest-cov pytest-mock
   ```

4. **Create your .env file**
   ```bash
   cp .env.example .env
   # Edit .env with your local development settings
   ```

5. **Verify installation**
   ```bash
   python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
   ```

### IDE Setup (VS Code)

1. **Install Python extension**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Python" and install the official Microsoft extension

2. **Configure VS Code settings** (.vscode/settings.json)
   ```json
   {
     "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": true,
     "python.formatting.provider": "black",
     "editor.formatOnSave": true,
     "editor.rulers": [88],
     "[python]": {
       "editor.defaultFormatter": "ms-python.python",
       "editor.formatOnSave": true
     }
   }
   ```

3. **Install additional tools**
   ```bash
   pip install black flake8 pylint mypy
   ```

## Project Architecture

### Layered Architecture

The project follows a clean, layered architecture:

```
┌─────────────────────────────────────┐
│        HTTP Layer (FastAPI)         │  Main entry point
├─────────────────────────────────────┤
│      Routers (Request Handlers)     │  /tags, /generate, /chat, /health
├─────────────────────────────────────┤
│      Services (Business Logic)      │  ModelAggregator, OllamaClient
├─────────────────────────────────────┤
│     Data Models (Pydantic)          │  Request/Response schemas
├─────────────────────────────────────┤
│   Configuration & Utils             │  Settings, logging, helpers
└─────────────────────────────────────┘
```

### Data Flow

```
Client Request
    ↓
FastAPI Router (HTTP parsing)
    ↓
Service Method (business logic)
    ↓
OllamaClient (HTTP communication)
    ↓
Ollama Backend Instance
    ↓
Response Back Through Stack
    ↓
Client Response
```

## Code Structure

### Directory Overview

```
src/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── models.py            # Pydantic data models
├── routers/
│   ├── __init__.py
│   ├── tags.py          # GET /tags endpoint
│   ├── generate.py      # POST /api/generate endpoint
│   ├── chat.py          # POST /api/chat/completions endpoint
│   └── health.py        # GET /health endpoint
├── services/
│   ├── __init__.py
│   ├── ollama_client.py # HTTP client for Ollama communication
│   └── model_aggregator.py # Model aggregation & routing logic
└── utils/
    ├── __init__.py
    └── logger.py        # Logging utilities (to be implemented)

tests/
├── __init__.py
├── test_structure.py    # Basic structural tests
├── test_model_aggregator.py
├── test_ollama_client.py
├── test_tags.py
├── test_generate.py
├── test_chat.py
└── test_health.py
```

### Key Files Explained

**src/main.py**
- FastAPI application instance
- Application lifespan management (startup/shutdown)
- Router registration
- CORS middleware configuration

**src/config.py**
- Settings class with environment variable validation
- Default values for all configuration options
- Type-safe configuration using Pydantic Settings

**src/models.py**
- Pydantic models for all request/response types
- Data validation and serialization
- Type hints for IDE support

**src/services/ollama_client.py**
- Async HTTP client for communicating with Ollama instances
- Methods for all Ollama API endpoints
- Error handling and retry logic

**src/services/model_aggregator.py**
- Aggregates models from both local and remote Ollama instances
- Handles prefix addition/removal
- Implements model caching
- Routes requests to appropriate backend

## Development Workflow

### Running the Application

**Development mode (with auto-reload):**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**With different settings:**
```bash
# Verbose logging
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Multiple workers
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Making Code Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature-name
   ```

2. **Make your changes**
   - Edit files in the src/ directory
   - Follow code style guidelines (see below)
   - Add appropriate comments and docstrings

3. **Test your changes**
   ```bash
   pytest tests/ -v
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/my-feature-name
   ```

### Using Docker for Development

1. **Build the Docker image**
   ```bash
   docker build -t ollama-wrapper:dev .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **View logs**
   ```bash
   docker-compose logs -f gateway
   ```

4. **Execute commands in container**
   ```bash
   docker-compose exec gateway bash
   ```

## Testing

### Test Structure

Each module should have a corresponding test file in the `tests/` directory:

- `src/services/model_aggregator.py` → `tests/test_model_aggregator.py`
- `src/routers/tags.py` → `tests/test_tags.py`

### Writing Tests

**Example test structure:**
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.services.model_aggregator import ModelAggregator

@pytest.mark.asyncio
async def test_get_all_models():
    """Test getting all models from both backends."""
    aggregator = ModelAggregator()
    
    # Mock the client methods
    with patch.object(aggregator, 'get_remote_models', new_callable=AsyncMock) as mock_remote:
        with patch.object(aggregator, 'get_local_models', new_callable=AsyncMock) as mock_local:
            mock_remote.return_value = [...]
            mock_local.return_value = [...]
            
            result = await aggregator.get_all_models()
            
            assert len(result) == 2
            mock_remote.assert_called_once()
            mock_local.assert_called_once()
```

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run with verbose output:**
```bash
pytest -v
```

**Run specific test file:**
```bash
pytest tests/test_tags.py -v
```

**Run with coverage report:**
```bash
pytest --cov=src --cov-report=html
```

**Run only fast tests:**
```bash
pytest -m "not slow"
```

**Run with debugging output:**
```bash
pytest --log-cli-level=DEBUG
```

### Test Coverage Goals

- Aim for >80% code coverage
- 100% coverage for critical paths (routing, model selection)
- Use `pytest-cov` to measure coverage

```bash
pytest --cov=src --cov-report=term-missing
```

## Debugging

### Using print() and logging

**Simple logging:**
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Model list updated")
logger.error("Failed to connect to remote server")
```

**Enable debug logging:**
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or run with debug flag
uvicorn src.main:app --reload --log-level debug
```

### Using VS Code Debugger

1. **Create .vscode/launch.json:**
   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: FastAPI",
         "type": "python",
         "request": "launch",
         "module": "uvicorn",
         "args": ["src.main:app", "--reload"],
         "jinja": true,
         "justMyCode": true
       }
     ]
   }
   ```

2. **Set breakpoints** in your code (click on line number)
3. **Press F5** to start debugging
4. **Use debug console** to inspect variables

### Using pdb (Python Debugger)

Add to your code:
```python
import pdb; pdb.set_trace()
```

Then run the application and interact with the debugger:
```bash
# Commands:
n       # Next line
s       # Step into function
c       # Continue
p var   # Print variable
l       # List current code
h       # Help
```

### Testing with Mock Ollama Servers

For local testing without real Ollama instances:

```python
from unittest.mock import AsyncMock, patch

async def test_with_mock_ollama():
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama2', 'size': 3000000000}
            ]
        }
        mock_get.return_value = mock_response
        
        # Your test code here
        pass
```

## Code Style & Standards

### Python Style Guide (PEP 8)

- 4 spaces for indentation
- Maximum line length: 88 characters
- Use type hints for all functions

### Naming Conventions

```python
# Classes: PascalCase
class ModelAggregator:
    pass

# Functions/methods: snake_case
def get_all_models():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_TIMEOUT = 300

# Private methods/attributes: _leading_underscore
def _internal_method():
    pass
```

### Type Hints

Always include type hints:
```python
# Good
async def fetch_models(model_id: str) -> List[ModelInfo]:
    pass

# Bad
async def fetch_models(model_id):
    pass
```

### Docstrings

Use Google-style docstrings:
```python
def get_all_models(self) -> List[ModelInfo]:
    """
    Get all models from both local and remote Ollama instances.
    
    The method fetches models concurrently from both backends and combines
    them with appropriate prefixes for identification.
    
    Returns:
        List[ModelInfo]: List of all available models with prefixes.
        
    Raises:
        ConnectionError: If unable to reach either Ollama instance.
    """
    pass
```

### Formatting

**Auto-format with Black:**
```bash
black src/ tests/
```

**Check style with flake8:**
```bash
flake8 src/ tests/
```

**Type checking with mypy:**
```bash
mypy src/
```

## Common Development Tasks

### Adding a New Endpoint

1. **Define the model** (src/models.py):
   ```python
   class MyRequest(BaseModel):
       param1: str
       param2: int
   ```

2. **Create the router** (src/routers/my_feature.py):
   ```python
   from fastapi import APIRouter
   
   router = APIRouter()
   
   @router.post("/my-endpoint")
   async def my_endpoint(request: MyRequest):
       return {"message": "success"}
   ```

3. **Register in main.py**:
   ```python
   from src.routers import my_feature
   app.include_router(my_feature.router, prefix="/api")
   ```

4. **Add tests** (tests/test_my_feature.py):
   ```python
   @pytest.mark.asyncio
   async def test_my_endpoint():
       # Test implementation
       pass
   ```

### Adding a New Configuration Option

1. **Update config.py**:
   ```python
   class Settings(BaseSettings):
       my_new_option: str = Field(
           default="default_value",
           description="Description of the option"
       )
   ```

2. **Update .env.example**:
   ```
   MY_NEW_OPTION=default_value
   ```

3. **Use in code**:
   ```python
   from src.config import settings
   value = settings.my_new_option
   ```

### Adding a New Service

1. **Create the service class** (src/services/my_service.py):
   ```python
   class MyService:
       def __init__(self):
           self.client = OllamaClient(...)
       
       async def my_method(self):
           pass
   ```

2. **Add tests** (tests/test_my_service.py)
3. **Use in routers** by importing and calling

### Updating Dependencies

```bash
# Add new dependency
pip install package-name
pip freeze > requirements.txt

# Update dependency
pip install --upgrade package-name
pip freeze > requirements.txt

# Remove dependency
pip uninstall package-name
pip freeze > requirements.txt
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
```bash
# Ensure PYTHONPATH includes the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root with -m flag
python -m pytest tests/

# Or reinstall in editable mode
pip install -e .
```

### Async/Await Errors

**Problem:** `RuntimeError: no running event loop`

**Solution:** Use `pytest-asyncio` and mark async tests:
```python
@pytest.mark.asyncio
async def test_async_function():
    pass
```

### Port Already in Use

**Problem:** `OSError: [Errno 48] Address already in use`

**Solutions:**
```bash
# Linux/Mac: Find and kill process
lsof -i :8000
kill -9 <PID>

# Windows: Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
uvicorn src.main:app --port 8001
```

### Environment Variable Not Loaded

**Problem:** `ValidationError: Configuration value not set`

**Solutions:**
```bash
# Verify .env file exists and is correct
cat .env

# Reload environment
source venv/bin/activate

# Or explicitly set variables
export REMOTE_OLLAMA_URL=http://192.168.1.155:11434
```

### Test Timeouts

**Problem:** Tests hanging or taking too long

**Solutions:**
```bash
# Set timeout limit
pytest --timeout=10

# Run only fast tests
pytest -m "not slow"

# Use specific test
pytest tests/test_specific.py::test_function
```

## Performance Tips

### During Development

1. **Use --reload flag** for auto-restart on changes
2. **Run only relevant tests** to save time
3. **Use fast mode** for unit tests
4. **Cache external API responses** during development

### For Production

1. **Use multiple workers** (`--workers 4`)
2. **Enable caching** in model aggregator
3. **Monitor logs** for performance issues
4. **Profile code** with cProfile if needed

```bash
# Profile with cProfile
python -m cProfile -s cumtime src/main.py
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python asyncio Guide](https://docs.python.org/3/library/asyncio.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Git Workflow Guide](https://git-scm.com/docs)

## Getting Help

1. Check existing documentation
2. Review similar code in the repository
3. Check Ollama API documentation
4. Open an issue with details
5. Ask in discussions or comments

---

**Last Updated:** 2024  
**Status:** Active Development