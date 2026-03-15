# Ollama Proxy Server with GitHub Models

A lightweight Python proxy that:

- Exposes every model from the remote Ollama server at `192.168.1.155` with a `155 | ` prefix
- Exposes GitHub Models (via the Azure AI Inference SDK) with a `GH | ` prefix
- Presents a unified Ollama-compatible API **and** an OpenAI-compatible `/v1/` surface so any client works without extra configuration

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your GitHub token

Create a `.env` file in the project root (see `.env.example`):

```
GITHUB_TOKEN=github_pat_...
```

The token needs the `models:read` permission scope.  
Get one at: https://github.com/settings/tokens

### 3. Run

```bash
python proxy.py
```

The proxy starts on **`http://localhost:5000`**.

---

## Continue.dev Configuration

In your Continue.dev `config.json`, point the provider at the proxy using the **OpenAI provider** type with the `/v1` base:

```json
{
  "models": [
    {
      "title": "GH | claude-haiku-4-5",
      "provider": "openai",
      "model": "GH | claude-haiku-4-5",
      "apiBase": "http://127.0.0.1:5000/v1/",
      "apiKey": "proxy"
    },
    {
      "title": "GH | gpt-4.1",
      "provider": "openai",
      "model": "GH | gpt-4.1",
      "apiBase": "http://127.0.0.1:5000/v1/",
      "apiKey": "proxy"
    },
    {
      "title": "155 | llama3.2",
      "provider": "openai",
      "model": "155 | llama3.2",
      "apiBase": "http://127.0.0.1:5000/v1/",
      "apiKey": "proxy"
    }
  ]
}
```

> **Note:** The `apiKey` value is ignored by the proxy but must be non-empty for Continue.dev to accept the config.

---

## Available Models

### Standard GitHub Models (`GH | ` prefix)

Available with a regular GitHub PAT (`GITHUB_TOKEN`). Confirmed working via `https://models.inference.ai.azure.com`:

| Display name | GitHub SDK model ID |
|---|---|
| `GH \| gpt-4o` | `gpt-4o` |
| `GH \| gpt-4o-mini` | `gpt-4o-mini` |
| `GH \| gpt-4.1` | `gpt-4.1` |
| `GH \| gpt-4.1-mini` | `gpt-4.1-mini` |
| `GH \| Meta-Llama-3.1-405B-Instruct` | `Meta-Llama-3.1-405B-Instruct` |
| `GH \| Meta-Llama-3.1-8B-Instruct` | `Meta-Llama-3.1-8B-Instruct` |

> To see exactly which models your token can access:
> ```bash
> curl -H "Authorization: Bearer $GITHUB_TOKEN" https://models.inference.ai.azure.com/models
> ```

### GitHub Copilot Models (`GC | ` prefix)

Requires a Copilot Pro / Business / Enterprise token (`GITHUB_TOKEN_COPILOT`). These models are hidden from the model list until `GITHUB_TOKEN_COPILOT` is set in your `.env`:

| Display name | GitHub SDK model ID |
|---|---|
| `GC \| claude-haiku-4-5` | `claude-haiku-4-5` |
| `GC \| claude-sonnet-4-5` | `claude-sonnet-4-5` |
| `GC \| claude-opus-4-5` | `claude-opus-4-5` |
| `GC \| gemini-2.0-flash` | `gemini-2.0-flash` |
| `GC \| grok-3` | `grok-3` |
| `GC \| grok-3-mini` | `grok-3-mini` |

### Remote Ollama Models (`155 | ` prefix)

Any model installed on the Ollama server at `192.168.1.155` is automatically listed with the `155 | ` prefix, e.g. `155 | llama3.2`, `155 | mistral`.

---

## API Endpoints

### OpenAI-compatible (recommended for Continue.dev)

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List all models in OpenAI format |
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |

### Ollama-native

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/tags` | List all models in Ollama format |
| `POST` | `/api/show` | Model metadata |
| `POST` | `/api/chat` | Ollama-style chat (NDJSON) |
| `POST` | `/api/generate` | Ollama-style generate (NDJSON) |
| `POST` | `/api/chat/completions` | OpenAI-compat chat (via Ollama path) |
| `POST` | `/api/embeddings` | Embeddings (legacy alias) |
| `POST` | `/api/embed` | Embeddings |
| `POST` | `/api/pull` | Pull a model on the remote server |
| `GET` | `/api/ps` | List running models |
| `DELETE` | `/api/delete` | Delete a model on the remote server |
| `POST` | `/api/copy` | Copy a model on the remote server |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Reports proxy health and remote Ollama status |

---

## Configuration

All configuration is via environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | *(required for `GH \|` models)* | Standard GitHub PAT — create at github.com/settings/tokens |
| `GITHUB_TOKEN_COPILOT` | *(optional, for `GC \|` models)* | Copilot Pro/Business/Enterprise PAT — unlocks Claude, Gemini, Grok |
| `REMOTE_OLLAMA_URL` | `http://192.168.1.155:11434` | Remote Ollama server address |
| `PORT` | `5000` | Port the proxy listens on |

---

## How It Works

1. **`GET /v1/models` or `GET /api/tags`** — the proxy fetches the live model list from `192.168.1.155`, prefixes each name with `155 | `, appends the GitHub model catalogue (each prefixed `GH | `), and returns the combined list.

2. **Request with `155 | <model>`** — the proxy strips the prefix and forwards the request verbatim to `http://192.168.1.155:11434`.

3. **Request with `GH | <model>`** — the proxy looks up the GitHub SDK model ID, calls `https://models.inference.ai.azure.com` via the `azure-ai-inference` SDK using your `GITHUB_TOKEN`, and translates the response back into Ollama or OpenAI format depending on which endpoint was called.

4. **Streaming** — both Ollama NDJSON streaming (`/api/chat`, `/api/generate`) and OpenAI SSE streaming (`/v1/chat/completions`) are fully supported.

---

## Troubleshooting

### `GITHUB_TOKEN is not configured`
Add `GITHUB_TOKEN=github_pat_...` to your `.env` file and restart the proxy.

### `Model not found` for a GitHub model
Check that the model name in your client matches exactly (including the `GH | ` prefix and dashes, not dots). The proxy also accepts bare names with dots, e.g. `claude-haiku-4.5` is normalised to `GH | claude-haiku-4-5` automatically.

### Remote Ollama models not showing
Verify the remote server is reachable:
```bash
curl http://192.168.1.155:11434/api/tags
```
Update `REMOTE_OLLAMA_URL` in `.env` if your server is at a different address.

### Port already in use
Set `PORT=<number>` in your `.env` file.

---

## License

MIT