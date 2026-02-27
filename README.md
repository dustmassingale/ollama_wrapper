# Ollama Wrapper Gateway

A gateway proxy service that aggregates models from multiple Ollama instances, providing a unified interface to access local and remote models with clear source identification.

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
│   │   ├── ollama_client.py    # HTTP client for Ollama instances
│   │   └── model_aggregator.py # Aggregates models from both sources
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── test_tags.py            # Tests for /tags endpoint
│   ├── test_generate.py        # Tests for /api/generate
│   ├── test_chat.py            # Tests for /api/chat
│   ├── test_model_aggregator.py # Tests for model aggregation
│   └── test_routing.py         # Tests for request routing
├── .env.example
├── requirements.txt
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose for deployment
├── README.md
└── .gitignore
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

# Model Prefix Configuration
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

**Response:** Proxied directly from the appropriate Ollama instance

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
  "stream": false
}
```

**Response:** Chat completion from the appropriate Ollama instance

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

## Implementation Progress

### ✅ Phase 1: Project Setup - COMPLETED
- [x] Initialize project structure with all directories
- [x] Create virtual environment and install dependencies
- [x] Set up FastAPI application skeleton with proper lifespan management
- [x] Configure environment variables and settings management (config.py)
- [x] Create Pydantic models for all request/response types
- [x] Set up all routers (/tags, /generate, /chat, /health)
- [x] Create .env.example template with all configuration options
- [x] Initialize Git repository with meaningful commits
- [x] Project dependencies installed and verified

**Status:** All files created and committed to Git. Core structure in place.

### 🔄 Phase 2: Core Services Implementation - IN PROGRESS
- [x] Created HTTP client class (OllamaClient) for API communication
- [x] Implemented ModelAggregator service with concurrent model fetching
- [x] Set up caching mechanism for model lists
- [x] Create connection checking methods
- [x] Implement model prefix stripping and routing logic
- [ ] Add retry logic with exponential backoff for failed requests
- [ ] Implement circuit breaker pattern for resilience
- [ ] Add comprehensive logging and error tracking
- [ ] Test all service methods with mock Ollama instances

**Current Focus:** Core services are scaffolded but need testing and refinement.

### 📋 Phase 3: Request Routing & Proxying - NOT STARTED
**Tasks:**
- [ ] Implement request parsing to extract model prefix from client requests
- [ ] Route requests to correct backend based on prefix (155- for remote, LOC- for local)
- [ ] Strip prefix from model name before forwarding to backend
- [ ] Implement streaming response handling for /generate and /chat endpoints
- [ ] Add request body validation and error responses
- [ ] Handle edge cases (missing model, invalid prefix, etc.)
- [ ] Implement proper HTTP header forwarding and modification
- [ ] Test routing with actual Ollama instances on 192.168.1.155

**Implementation Time Estimate:** 2-3 hours

### 🛡️ Phase 4: Error Handling & Resilience - NOT STARTED
**Tasks:**
- [ ] Implement connection retry logic with exponential backoff
- [ ] Add circuit breaker pattern to prevent cascading failures
- [ ] Implement timeout handling for long-running requests
- [ ] Graceful degradation when one Ollama instance is unavailable
- [ ] Add detailed error messages and logging throughout
- [ ] Create custom exception classes for different error scenarios
- [ ] Implement request/response validation error handling
- [ ] Add metrics collection for monitoring (hits, failures, latency)

**Implementation Time Estimate:** 2-3 hours

### 🧪 Phase 5: Testing - NOT STARTED
**Test Files to Create:**

1. **test_model_aggregator.py**
   - Test model fetching from both instances
   - Test model caching mechanism
   - Test prefix addition/stripping
   - Test connection checking

2. **test_routing.py**
   - Test prefix parsing
   - Test correct backend selection
   - Test model name normalization
   - Test invalid model name handling

3. **test_tags.py**
   - Test /tags endpoint returns all models
   - Test model prefix in responses
   - Test error handling when backends unavailable

4. **test_generate.py**
   - Test /api/generate with remote models
   - Test /api/generate with local models
   - Test streaming responses
   - Test invalid model errors

5. **test_chat.py**
   - Test /api/chat with remote models
   - Test /api/chat with local models
   - Test streaming chat responses
   - Test message format validation

6. **test_health.py**
   - Test /health endpoint status responses
   - Test backend connectivity detection
   - Test partial availability scenarios

**Implementation Time Estimate:** 3-4 hours

### 📦 Phase 6: Deployment Configuration - NOT STARTED
**Files to Create:**

1. **Dockerfile**
   - Base image: python:3.11-slim
   - Install dependencies
   - Copy application code
   - Set up working directory
   - Expose port 8000
   - Default command to run uvicorn

2. **docker-compose.yml**
   - Define ollama_wrapper service
   - Port mappings
   - Volume mounts for .env
   - Network configuration
   - Resource limits

3. **systemd service file** (optional)
   - Service definition for Linux deployment
   - Auto-restart configuration
   - User/group configuration

4. **Nginx reverse proxy config** (optional)
   - SSL/TLS termination
   - Load balancing if multiple gateway instances
   - Request logging

**Implementation Time Estimate:** 1-2 hours

### 📚 Phase 7: Documentation & Polish - NOT STARTED
**Tasks:**
- [ ] Add docstrings to all functions and classes
- [ ] Create comprehensive API documentation
- [ ] Add usage examples for common scenarios
- [ ] Create troubleshooting guide with common issues
- [ ] Add architecture decision records (ADR)
- [ ] Document deployment procedures
- [ ] Create development setup guide
- [ ] Add performance tuning recommendations
- [ ] Generate API docs from FastAPI (auto-generated at /docs)

**Implementation Time Estimate:** 2-3 hours

### 🚀 Phase 8: Production Readiness - NOT STARTED
**Tasks:**
- [ ] Add security headers and CORS configuration
- [ ] Implement request rate limiting
- [ ] Add authentication/authorization layer (optional)
- [ ] Set up comprehensive logging infrastructure
- [ ] Add metrics and monitoring (Prometheus integration)
- [ ] Create health check and readiness probes for K8s
- [ ] Performance testing and optimization
- [ ] Load testing with multiple concurrent requests
- [ ] Security audit of dependencies

**Implementation Time Estimate:** 4-6 hours

## Getting Started

### Prerequisites
- Python 3.9+
- Ollama server running on 192.168.1.155 with models available
- Local Ollama server (optional, for local models)
- pip and venv for Python package management

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
# Edit .env with your actual server addresses if different from defaults
```

5. Run the gateway
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The gateway will be available at `http://localhost:8000` with interactive API docs at `http://localhost:8000/docs`

## Development Workflow

### Current State
The project has a complete scaffolding with all file structures in place. The main components have been created:
- Configuration system working
- Data models defined
- Routers set up but need route handler implementation
- Services created with basic methods

### Next Immediate Steps
1. **Debug and fix imports** - Ensure all modules import correctly
2. **Implement router handlers** - Fill in the actual endpoint logic
3. **Test with mock Ollama** - Verify communication works
4. **Add error handling** - Implement proper exception handling
5. **Create integration tests** - Test with real Ollama instances

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src
```

Run specific test file:
```bash
pytest tests/test_tags.py -v
```

Run with verbose output:
```bash
pytest -v
```

## Docker Deployment

### Build the Docker image
```bash
docker build -t ollama-wrapper .
```

### Run standalone
```bash
docker run -d \
  --name ollama-wrapper \
  -p 8000:8000 \
  -e REMOTE_OLLAMA_URL=http://192.168.1.155:11434 \
  -e LOCAL_OLLAMA_URL=http://localhost:11434 \
  ollama-wrapper
```

### Run with docker-compose
```bash
docker-compose up -d
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
    "prompt": "Why is the sky blue?"
  }'
```

### Generate text using local model
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LOC-mistral",
    "prompt": "Explain quantum computing"
  }'
```

### Chat with a remote model
```bash
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "155-llama3.1:8b",
    "messages": [
      {
        "role": "user",
        "content": "Hello! How are you?"
      }
    ]
  }'
```

### Check gateway health
```bash
curl http://localhost:8000/health
```

### Access interactive API documentation
Open your browser and navigate to: `http://localhost:8000/docs`

## Troubleshooting

### Remote server not responding
- Verify 192.168.1.155 is reachable: `ping 192.168.1.155`
- Check Ollama is running on remote server: `curl http://192.168.1.155:11434/api/tags`
- Verify port 11434 is accessible from your network
- Check firewall rules on both machines
- Ensure network routing is configured correctly

### Models not appearing in /tags
- Ensure both Ollama servers are running
- Check environment variables in .env match your setup
- Review gateway logs for connection errors: `tail -f logs/gateway.log`
- Verify models are actually available on the servers: `curl http://192.168.1.155:11434/api/tags`
- Check if the servers have any authentication enabled

### Slow response times
- Check network latency to 192.168.1.155: `ping -c 5 192.168.1.155`
- Monitor CPU/memory usage on both servers
- Consider increasing cache timeout in .env
- Check if model needs to be pulled/loaded on first use
- Monitor gateway logs for slow requests
- Check if model is being offloaded to disk (runs very slowly)

### Connection timeouts
- Increase REQUEST_TIMEOUT in .env if requests are legitimately slow
- Check if the remote server is overloaded
- Verify no packet loss on the network
- Check firewall/NAT configuration if behind corporate firewall
- Try accessing the remote server directly to isolate the issue

### Import errors when running
- Ensure all files are created in the correct directories
- Verify Python path includes the src directory
- Check that __init__.py files exist in all packages
- Try reinstalling dependencies: `pip install -r requirements.txt --force-reinstall`

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Language** | Python | Quick development, excellent HTTP/async support |
| **Framework** | FastAPI | Async, modern, auto-documentation with Swagger UI |
| **Prefix Style** | `155-` / `LOC-` | Clear, explicit source identification |
| **Caching** | In-memory with TTL | Reduces load on backends, simple to implement |
| **Error Handling** | Graceful degradation | Maximize availability when one backend fails |
| **Streaming** | Direct proxy | Low latency, minimal memory usage for large responses |
| **HTTP Client** | httpx | Async support, connection pooling, follows redirects |
| **Async/Await** | Throughout | Non-blocking I/O, better concurrency handling |

## Request Flow

```
1. Client Request
   ↓
2. Parse Model Name (e.g., "155-llama3.1:8b")
   ↓
3. Determine Source (Check prefix: 155- = Remote, LOC- = Local)
   ↓
4. Strip Prefix (Convert "155-llama3.1:8b" to "llama3.1:8b")
   ↓
5. Route to Correct Backend
   ├─→ If Remote (155-): Send to 192.168.1.155:11434
   └─→ If Local (LOC-): Send to localhost:11434
   ↓
6. Proxy Request with Original Parameters
   ↓
7. Receive and Stream Response
   ↓
8. Return to Client
   ↓
9. Handle Errors & Log Activity
```

## Performance Considerations

- **Model Caching**: Model lists are cached for 300 seconds (configurable) to reduce backend queries
- **Connection Pooling**: HTTPX maintains connection pools to both Ollama instances
- **Streaming**: Responses are streamed directly without buffering large payloads
- **Concurrent Requests**: Async design allows handling multiple simultaneous requests
- **Timeout Configuration**: Defaults to 300 seconds for long-running model operations

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes and test thoroughly
3. Add tests for new functionality
4. Update documentation as needed
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your branch: `git push origin feature/my-feature`
7. Submit a Pull Request with detailed description

## License

MIT License - See LICENSE file for details

## Support & Issues

For issues, questions, or feature requests:
- Check this README's troubleshooting section first
- Review the API documentation at `/docs` endpoint (when running)
- Check application logs for detailed error messages
- Create an issue on the repository with detailed information

## Future Roadmap

### Short Term (Next Features)
- [ ] Streaming request/response handling optimization
- [ ] Advanced logging and request tracing
- [ ] Configuration hot-reload without restart
- [ ] WebSocket support for real-time model streaming

### Medium Term
- [ ] Redis-based caching for distributed deployments
- [ ] Load balancing across multiple gateway instances
- [ ] Authentication and authorization system (API keys, OAuth)
- [ ] Rate limiting per client/model
- [ ] Request queuing and priority levels

### Long Term
- [ ] Metrics collection and Prometheus integration
- [ ] Web UI for model and gateway management
- [ ] Support for other model serving platforms (vLLM, TGI, etc.)
- [ ] Kubernetes deployment with Helm charts
- [ ] Multi-region federation support
- [ ] Advanced caching strategies (semantic similarity, etc.)

## Architecture Notes

The gateway implements a clean separation of concerns:

- **Routers**: Handle HTTP request/response serialization
- **Services**: Implement business logic (aggregation, routing)
- **Models**: Define data structures with Pydantic validation
- **Config**: Manage all configuration and environment variables

This architecture makes the system:
- **Testable**: Each layer can be tested independently
- **Maintainable**: Clear responsibilities for each component
- **Scalable**: Easy to add new endpoints or backends
- **Debuggable**: Clean separation makes debugging easier

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [HTTPX Documentation](https://www.python-httpx.org/)

---

**Last Updated**: After Phase 1 Implementation
**Current Version**: 0.1.0
**Status**: Active Development