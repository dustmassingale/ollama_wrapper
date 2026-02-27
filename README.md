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
│   │   ├── ollama_client.py    # Client for 192.168.1.155 Ollama
│   │   ├── local_client.py     # Client for local Ollama
│   │   └── model_aggregator.py # Aggregates models from both sources
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── test_tags.py
│   ├── test_generate.py
│   └── test_chat.py
├── .env.example
├── requirements.txt
├── README.md
└── docker-compose.yml (optional)
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
  "stream": false
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
  ]
}
```

**Response:** Streamed responses from the appropriate Ollama instance

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

## Implementation Plan

### Phase 1: Project Setup
- [x] Initialize project structure
- [ ] Set up FastAPI skeleton
- [ ] Configure dependencies (FastAPI, uvicorn, httpx, pydantic)
- [ ] Create configuration management system

### Phase 2: Core Components
- [ ] Implement model aggregator service
- [ ] Create Ollama HTTP clients (remote and local)
- [ ] Build Pydantic models for requests/responses
- [ ] Set up async HTTP communication

### Phase 3: API Endpoints
- [ ] Implement `/tags` endpoint
- [ ] Implement `/api/generate` endpoint with streaming
- [ ] Implement `/api/chat/completions` endpoint
- [ ] Implement `/health` endpoint

### Phase 4: Request Routing
- [ ] Parse model prefixes (155- and LOC-)
- [ ] Route requests to appropriate backend
- [ ] Handle model name normalization
- [ ] Implement fallback logic

### Phase 5: Error Handling & Resilience
- [ ] Connection retry logic with exponential backoff
- [ ] Timeout handling for long-running requests
- [ ] Graceful degradation when one server is unavailable
- [ ] Comprehensive error logging

### Phase 6: Testing
- [ ] Unit tests for model aggregation
- [ ] Unit tests for prefix parsing
- [ ] Integration tests for full request flows
- [ ] Error scenario testing

### Phase 7: Deployment
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Create deployment documentation
- [ ] Add example systemd service configuration

### Phase 8: Documentation
- [ ] API documentation (auto-generated by FastAPI)
- [ ] Setup instructions
- [ ] Configuration guide
- [ ] Example usage scenarios
- [ ] Troubleshooting guide

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Language** | Python | Quick development, excellent HTTP support |
| **Framework** | FastAPI | Async, modern, auto-documentation with Swagger UI |
| **Prefix Style** | `155-` / `LOC-` | Clear, explicit source identification |
| **Caching** | In-memory with TTL | Reduces load on backends |
| **Error Handling** | Graceful degradation | Maximize availability |
| **Streaming** | Direct proxy | Low latency, minimal memory usage |
| **HTTP Client** | httpx | Async support, connection pooling |

## Request Flow

1. Client sends request with model name (e.g., `155-llama3.1:8b`)
2. Gateway parses model name and determines source (155 or LOC prefix)
3. Prefix is stripped from model name (e.g., `llama3.1:8b`)
4. Request is routed to appropriate backend (remote or local)
5. Request is proxied to backend Ollama instance
6. Response is streamed back to client
7. Errors are handled with appropriate fallbacks

## Getting Started

### Prerequisites
- Python 3.9+
- Ollama server running on 192.168.1.155
- Local Ollama server (optional, for local models)

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
# Edit .env with your configuration
```

5. Run the gateway
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The gateway will be available at `http://localhost:8000` with interactive API docs at `http://localhost:8000/docs`

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src
```

## Docker Deployment

Build the Docker image:
```bash
docker build -t ollama-wrapper .
```

Run with docker-compose:
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

### Chat with a local model
```bash
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LOC-mistral",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

### Check gateway health
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Remote server not responding
- Verify 192.168.1.155 is reachable: `ping 192.168.1.155`
- Check Ollama is running on remote server
- Verify port 11434 is accessible
- Check firewall rules

### Models not appearing in /tags
- Ensure Ollama servers are running
- Check environment variables are correct
- Review gateway logs for connection errors

### Slow response times
- Check network latency to 192.168.1.155
- Monitor CPU/memory usage on both servers
- Consider increasing cache timeout
- Check if model needs to be pulled/loaded

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for new functionality
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `/docs` endpoint
- Check logs for detailed error messages

## Roadmap

- [ ] Load balancing across multiple instances
- [ ] Advanced caching strategies (Redis)
- [ ] Authentication and authorization
- [ ] Rate limiting per client/model
- [ ] Metrics and monitoring (Prometheus)
- [ ] Web UI for model management
- [ ] Support for other model serving platforms