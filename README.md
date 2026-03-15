# Ollama Proxy Server with GitHub Copilot Models

A simple Python proxy that routes requests to your Ollama server at 192.168.1.155 and adds GitHub Copilot models for IDE integration.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python proxy.py
```

The proxy will start on `http://localhost:5000`

## Features

- ✅ Proxies all requests to Ollama server at 192.168.1.155:11434
- ✅ Adds GitHub Copilot models:
  - GPT-5 mini
  - Claude Haiku 4.5
  - Claude Opus 4.6
  - Claude Sonnet 4.6
  - Gemini 3 Flash (Preview)
  - Gemini 3.1 Pro (Preview)
  - GPT-5.3-Codex
  - GPT-5.4
  - Grok Code Fast 1

## API Endpoints

### GET /
Returns API information

### GET /api/tags
Lists all available models (Ollama + GitHub)

### GET /health
Health status of proxy and remote Ollama server

### POST /api/generate
Generate text using specified model
```json
{
  "model": "llama2",
  "prompt": "Hello world"
}
```

### POST /api/chat/completions
Chat completion endpoint
```json
{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

## Configuration

Edit `proxy.py` to change:
- `REMOTE_OLLAMA_URL` - Your Ollama server address (default: http://192.168.1.155:11434)
- `GITHUB_MODELS` - List of available GitHub models
- Port - Currently 5000, change in `app.run()` at bottom

## Usage Examples

### List all models
```bash
curl http://localhost:5000/api/tags
```

### Generate text
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "prompt": "Why is the sky blue?"}'
```

### Chat
```bash
curl -X POST http://localhost:5000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Health check
```bash
curl http://localhost:5000/health
```

## How It Works

1. **Local models** - Any request for a model that exists on your Ollama server is proxied directly
2. **GitHub models** - Requests for GitHub Copilot models return a placeholder response
3. **Listing** - `/api/tags` combines models from both sources

## Troubleshooting

### Remote server not responding
- Verify Ollama is running: `curl http://192.168.1.155:11434/api/tags`
- Check firewall allows access to port 11434
- Update `REMOTE_OLLAMA_URL` in proxy.py if your server is at a different address

### Models not showing up
- Make sure Ollama is running on 192.168.1.155
- Models may need to be pulled first: `ollama pull llama2`

### Port already in use
Change the port in proxy.py line at the bottom:
```python
app.run(host="0.0.0.0", port=5000, debug=True)  # Change 5000 to another port
```

## License

MIT