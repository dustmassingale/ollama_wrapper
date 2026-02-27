# Ollama Wrapper Gateway

A gateway proxy service that aggregates models from multiple Ollama instances, providing a unified interface to access local and remote models with clear source identification.

> **🎉 Phase 1 Complete!** See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for details on what has been completed.

## Overview

This project creates a centralized gateway that acts as a proxy for:
- **Remote Ollama Server** (192.168.1.155) - Ollama instance with llama3.1:8b and other models
- **Local Ollama Server** - Models running on the gateway machine

All models are exposed through a single API with prefixes for easy identification:
- `155-` for models running on 192.168.1.155
- `LOC-` for locally running models

## Architecture

```
┌─────────────────────────────────────┐
│   Ollama Wrapper Gateway            │
│   (Proxy/Aggregation Layer)         │
├─────────────────────────────────────┤
│  • /tags - List all available models│
│  • /api/generate - Proxy requests   │
│  • /api/chat - Proxy requests       │
│  • Health checks                    │
└─────────────────────────────────────┘
         ↓                    ↓
┌──────────────────┐  ┌──────────────────┐
│ 192.168.1.155    │  │  Local Machine   │
│ • Ollama Server  │  │ • Ollama Server  │
│ • Llama3.1:8b    │  │ • Other models   │
└──────────────────┘  └──────────────────┘
```

## Project Structure

```
ollama_wrapper/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Configuration management
│   ├── models.py               # Pydantic models for requests/responses
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── tags.py             # /tags endpoint
│   │   ├── generate.py         # /api/generate proxy
│   │   ├── chat.py             # /api/chat proxy
│   │   └── health.py           # Health check endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ollama_client.py    # HTTP clients for Ollama instances
│   │   └── model_aggregator.py # Aggregates models from both sources
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── test_structure.py        # Structural validation tests
│   ├── test_model_aggregator.py # Model aggregation tests
│   ├── test_ollama_client.py    # HTTP client tests
│   ├── test_tags.py             # Tags endpoint tests
│   ├── test_generate.py         # Generate endpoint tests
│   ├── test_chat.py             # Chat endpoint tests
│   └── test_health.py           # Health check tests
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── LICENSE
├── README.md
├── DEVELOPMENT.md
└── IMPLEMENTATION_SUMMARY.md
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Remote Ollama Server (192.168.1.155)
REMOTE_OLLAMA_URL=http://192.168.1.155:11434

# Local Ollama Server
LOCAL_OLLAMA_URL=http://localhost:11434

# Gateway Configuration
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000

# Performance Settings
CACHE_TIMEOUT=300
REQUEST_TIMEOUT=300

# Model prefix configuration
REMOTE_MODEL_PREFIX=155-
LOCAL_MODEL_PREFIX=LOC-
```

## API Endpoints

### GET /tags
Returns all available models from both local and remote servers with appropriate prefixes.

**Response:**
```json
{
  "models": [
    {
      "name": "155-llama3.1:8b",
      "modified_at": "2024-01-15T10:30:00Z",
      "size": 4700000000
    },
    {
      "name": "LOC-mistral",
      "modified_at": "2024-01-14T15:45:00Z",
      "size": 4100000000
    }
  ]
}
```

### POST /api/generate
Generates text based on a prompt using the specified model.

**Request:**
```json
{
  "model": "155-llama3.1:8b",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {}
}
```

**Response:** 
```json
{
  "model": "155-llama3.1:8b",
  "created_at": "2024-01-15T10:30:00Z",
  "response": "The sky appears blue because...",
  "done": true,
  "total_duration": 5000000000,
  "load_duration": 1000000000,
  "prompt_eval_count": 10,
  "eval_count": 50
}
```

### POST /api/chat/completions
Chat completion endpoint compatible with OpenAI-style requests.

**Request:**
```json
{
  "model": "155-llama3.1:8b",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "stream": false,
  "options": {}
}
```

**Response:** 
```json
{
  "model": "155-llama3.1:8b",
  "created_at": "2024-01-15T10:30:00Z",
  "response": "I'm doing well, thank you for asking!",
  "done": true
}
```

### GET /health
Returns the health status of the gateway and both Ollama instances.

**Response:**
```json
{
  "status": "healthy",
  "gateway": "ok",
  "remote_ollama": "ok",
  "local_ollama": "ok"
}
```

### GET /
Root endpoint providing information about the gateway.

**Response:**
```json
{
  "message": "Ollama Wrapper Gateway",
  "version": "0.1.0",
  "documentation": "/docs"
}
```

## Implementation Status

### ✅ Phase 1: Project Setup - COMPLETED
- [x] Initialize project structure
- [x] Set up FastAPI skeleton with lifespan management
- [x] Configure dependencies (FastAPI, uvicorn, httpx, pydantic)
- [x] Create configuration management system with pydantic-settings
- [x] Create Pydantic models for requests/responses
- [x] Create routers for all endpoints (tags, generate, chat, health)
- [x] Create HTTP client service (ollama_client.py)
- [x] Create model aggregator service (model_aggregator.py)
- [x] Initialize git repository and make initial commit
- [x] Create comprehensive documentation
- [x] Add Docker support with Dockerfile and docker-compose.yml
- [x] Add development guide (DEVELOPMENT.md)
- [x] Create implementation summary (IMPLEMENTATION_SUMMARY.md)

**Status:** All files created and committed to Git. Core structure in place.

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for complete details on Phase 1 completion.

### ⏳ Phase 2: Core Implementation - IN PROGRESS
- [ ] **Fix import issues** - Resolve any circular dependencies
- [ ] **Complete model aggregation** - Implement full model listing with prefix handling
- [ ] **Implement streaming support** - Handle streaming responses for generate/chat
- [ ] **Complete HTTP client methods** - Ensure all endpoints are properly proxied
- [ ] **Add request validation** - Validate model names before proxying
- [ ] **Implement prefix stripping** - Remove prefixes before sending to backends
- [ ] **Test basic connectivity** - Verify gateway can reach both Ollama servers

### ⏱️ Phase 3: Request Routing - PENDING
- [ ] Parse and validate model prefixes (155- and LOC-)
- [ ] Route requests to appropriate backend
- [ ] Handle model name normalization
- [ ] Implement fallback logic for unavailable servers
- [ ] Add detailed logging for request routing

### ⏱️ Phase 4: Error Handling & Resilience - PENDING
- [ ] Connection retry logic with exponential backoff
- [ ] Timeout handling for long-running requests
- [ ] Graceful degradation when one server is unavailable
- [ ] Comprehensive error logging
- [ ] Error response formatting
- [ ] Connection pooling optimization

### ⏱️ Phase 5: Testing - PENDING
- [ ] Unit tests for model aggregation
- [ ] Unit tests for prefix parsing and stripping
- [ ] Unit tests for HTTP client
- [ ] Integration tests for full request flows
- [ ] Error scenario testing
- [ ] Streaming response tests
- [ ] Health check tests
- [ ] Achieve >80% code coverage

### ⏱️ Phase 6: Deployment - PENDING
- [ ] Create Dockerfile with multi-stage build
- [ ] Create docker-compose.yml for local development
- [ ] Create docker-compose production configuration
- [ ] Create .gitignore file
- [ ] Add deployment documentation
- [ ] Add example systemd service configuration
- [ ] Add GitHub Actions CI/CD configuration

### ⏱️ Phase 7: Documentation - PENDING
- [ ] Complete API documentation (auto-generated by FastAPI at `/docs`)
- [ ] Create INSTALLATION.md with detailed setup instructions
- [ ] Create CONFIGURATION.md with all config options
- [ ] Create TROUBLESHOOTING.md with common issues
- [ ] Add architecture diagrams
- [ ] Add examples for each API endpoint
- [ ] Create CONTRIBUTING.md

### ⏱️ Phase 8: Production Hardening - PENDING
- [ ] Add authentication/authorization options
- [ ] Add rate limiting
- [ ] Add request/response logging
- [ ] Add metrics collection (Prometheus)
- [ ] Add monitoring alerts
- [ ] Performance optimization
- [ ] Security auditing

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Language** | Python 3.9+ | Quick development, excellent HTTP support |
| **Framework** | FastAPI | Async, modern, auto-documentation with Swagger UI |
| **Prefix Style** | `155-` / `LOC-` | Clear, explicit source identification |
| **Caching** | In-memory with TTL | Reduces load on backends, fast response |
| **Error Handling** | Graceful degradation | Maximize availability, serve available models |
| **Streaming** | Direct proxy via httpx | Low latency, minimal memory usage |
| **HTTP Client** | httpx (async) | Async support, connection pooling, modern |
| **Configuration** | Pydantic Settings | Type-safe, environment-based configuration |
| **API Format** | Ollama-compatible | Direct compatibility with Ollama API |

## Request Flow

1. Client sends request with prefixed model name (e.g., `155-llama3.1:8b`)
2. Gateway receives request at `/tags`, `/api/generate`, or `/api/chat/completions`
3. Router passes request to appropriate service method
4. Model aggregator/service determines source based on prefix:
   - `155-` → route to 192.168.1.155:11434
   - `LOC-` → route to local_ollama_url
5. Prefix is stripped from model name before sending to backend
6. HTTP client proxies request to backend Ollama instance
7. Response is returned (streamed if applicable)
8. Errors are caught and formatted appropriately

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Ollama server running on 192.168.1.155:11434 (optional but recommended)
- Local Ollama server on localhost:11434 (optional)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd ollama_wrapper
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment
```bash
cp .env.example .env
# Edit .env with your actual Ollama server addresses and configuration
```

5. Run the gateway
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The gateway will be available at `http://localhost:8000` with interactive API documentation at `http://localhost:8000/docs`

## Development

### Running the Application

**Development mode (with auto-reload):**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Production mode:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=html
```

**Run specific test file:**
```bash
pytest tests/test_tags.py -v
```

**Run with logging output:**
```bash
pytest --log-cli-level=DEBUG
```

## Docker Deployment

### Build the Docker image
```bash
docker build -t ollama-wrapper:latest .
```

### Run with Docker
```bash
docker run -p 8000:8000 \
  -e REMOTE_OLLAMA_URL=http://192.168.1.155:11434 \
  -e LOCAL_OLLAMA_URL=http://localhost:11434 \
  ollama-wrapper:latest
```

### Run with docker-compose
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f
```

## Example Usage

### List all available models
```bash
curl http://localhost:8000/tags
```

### Generate text using remote model
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "155-llama3.1:8b",
    "prompt": "Explain quantum computing in simple terms",
    "stream": false
  }'
```

### Stream text generation
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "155-llama3.1:8b",
    "prompt": "Write a short poem about technology",
    "stream": true
  }' | jq .
```

### Chat with a local model
```bash
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LOC-mistral",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant"
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "stream": false
  }'
```

### Check gateway health
```bash
curl http://localhost:8000/health
```

### Interactive API Documentation
Visit `http://localhost:8000/docs` for Swagger UI or `http://localhost:8000/redoc` for ReDoc

## Troubleshooting

### Remote server not responding
- Verify 192.168.1.155 is reachable: `ping 192.168.1.155`
- Check Ollama is running on remote server: `curl http://192.168.1.155:11434/api/tags`
- Verify port 11434 is accessible and not firewalled
- Check network connectivity and routing

### Models not appearing in /tags
- Ensure at least one Ollama server is running
- Check environment variables in `.env` match your setup
- Review gateway logs for connection errors
- Test Ollama servers directly to ensure they're responsive

### Connection timeout errors
- Increase `REQUEST_TIMEOUT` in `.env` (default: 300 seconds)
- Check network latency to 192.168.1.155
- Verify firewall is not blocking connections
- Ensure Ollama servers are not under heavy load

### Slow response times
- Check network latency to 192.168.1.155
- Monitor CPU/memory usage on both Ollama servers
- Check if models need to be loaded (first request is slower)
- Monitor gateway logs for slow operations
- Consider increasing `CACHE_TIMEOUT` for model list

### Import errors or syntax errors
- Ensure Python 3.9+ is being used: `python --version`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Clear Python cache: `find . -type d -name __pycache__ -exec rm -r {} +`

### Port already in use
- Change port in `.env` file
- Or kill existing process: `lsof -i :8000` (Linux/Mac) or `netstat -ano | findstr :8000` (Windows)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and add tests
4. Commit with clear messages: `git commit -m "Add feature: description"`
5. Push to your branch: `git push origin feature/your-feature-name`
6. Create a Pull Request with description of changes

### Code Style
- Follow PEP 8 conventions
- Use type hints for all functions
- Add docstrings to all public methods
- Run tests before submitting PR

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check the **Troubleshooting** section above
- Review the **API documentation** at `http://localhost:8000/docs`
- Check application logs for detailed error messages
- Review GitHub Issues for similar problems
- See [DEVELOPMENT.md](DEVELOPMENT.md) for development-specific help

## Project Roadmap

### Short Term (v0.2.0)
- [ ] Complete Phase 2-4 implementation
- [ ] Achieve 80%+ test coverage
- [ ] Full error handling and resilience
- [ ] Docker support

### Medium Term (v0.3.0)
- [ ] Authentication and authorization
- [ ] Rate limiting per client/model
- [ ] Advanced caching with Redis support
- [ ] Monitoring and metrics (Prometheus)

### Long Term (v1.0.0)
- [ ] Load balancing across multiple instances
- [ ] Web UI for model management
- [ ] Database for persistence
- [ ] Support for other model serving platforms
- [ ] Kubernetes deployment support
- [ ] Horizontal scaling support

## Version History

### v0.1.0 (Current)
- Initial project structure and setup
- FastAPI skeleton with all endpoint routers
- Configuration management
- Model aggregation service
- HTTP client for Ollama communication
- Health check endpoint
- Comprehensive README and documentation

## Contact & Support

For questions, feature requests, or bug reports, please:
- Open an issue on GitHub
- Check existing documentation and examples
- Review the API documentation at `/docs` endpoint

---

**Last Updated:** February 26, 2024  
**Status:** Phase 1 Complete ✅ | Phase 2+ Ready for Implementation 🚀
**Current Version:** 0.1.0  
**Documentation:** See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for Phase 1 completion details