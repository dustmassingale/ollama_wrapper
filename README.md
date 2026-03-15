# Ollama Proxy Server

A lightweight Python proxy that unifies **remote Ollama models**, **GitHub Models**, **GitHub Copilot Chat models**, and **AWS Bedrock models** behind a single Ollama-compatible API *and* an OpenAI-compatible `/v1/` surface — so any client (Continue.dev, Open WebUI, Cursor, etc.) works without extra configuration.

---

## Model Prefixes

| Prefix | Source | Auth |
|---|---|---|
| `GH \|` | GitHub Models via Azure AI Inference | GitHub PAT (`GITHUB_TOKEN`) |
| `GC \|` | GitHub Copilot Chat API | OAuth device flow (interactive login) |
| `BR \|` | AWS Bedrock (converse / invoke) | IAM access key |
| `155 \|` | Remote Ollama server at `REMOTE_OLLAMA_URL` | None |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

At minimum, set your GitHub PAT to enable `GH |` models:

```
GITHUB_TOKEN=github_pat_...
```

### 3. Run

```bash
python proxy.py
```

The proxy starts on **`http://localhost:5000`** (or the port set in `PORT`).

---

## Configuration

All configuration is via environment variables or a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | *(required for `GH \|`)* | GitHub PAT — create at [github.com/settings/tokens](https://github.com/settings/tokens). No special scopes needed. |
| `GITHUB_TOKEN_COPILOT` | *(set automatically)* | Copilot OAuth token. Do **not** set by hand — the proxy fills this in after device-flow login. |
| `AWS_ACCESS_KEY_ID` | *(optional)* | IAM access key ID. Required for `BR \|` models. |
| `AWS_SECRET_ACCESS_KEY` | *(optional)* | IAM secret access key. Required for `BR \|` models. |
| `AWS_BEDROCK_REGION` | `us-east-1` | AWS region with Bedrock model access enabled. |
| `REMOTE_OLLAMA_URL` | `http://192.168.1.155:11434` | Remote Ollama server URL. |
| `PORT` | `5000` | TCP port the proxy listens on. |
| `DISABLE_SSL_VERIFY` | `false` | Set to `true` to bypass SSL certificate verification. **See SSL note below.** |

### SSL Certificate Verification

Corporate and enterprise networks often inject a self-signed certificate into the TLS chain, causing errors like:

```
self signed certificate in certificate chain
```

To work around this, add the following to your `.env`:

```
DISABLE_SSL_VERIFY=true
```

> ⚠️ **WARNING:** Do NOT enable this in a production environment. It disables all SSL certificate validation, making connections vulnerable to man-in-the-middle attacks. Use only for local testing or as a temporary workaround while a proper CA bundle is arranged (e.g. setting `REQUESTS_CA_BUNDLE` to your corporate root certificate).

When this flag is active, the proxy logs a prominent warning at startup.

---

## GitHub Copilot Login (`GC |` models)

Copilot models (Claude, Gemini, Grok, GPT-4o, Llama, and more) use the GitHub Copilot Chat API and require an OAuth token obtained via browser login. Requires an active GitHub Copilot Individual or Business subscription.

**Steps:**

1. Start the proxy: `python proxy.py`
2. Open in your browser: [http://localhost:5000/auth/copilot](http://localhost:5000/auth/copilot)
3. The browser will open automatically to the GitHub device authorisation page
4. Enter the displayed code and approve the request
5. Check status at: [http://localhost:5000/auth/status](http://localhost:5000/auth/status)

The token is saved automatically to `.env` and survives restarts. `GC |` models appear in `/api/tags` immediately after login.

---

## AWS Bedrock Setup (`BR |` models)

`BR |` models are only advertised if they are **not** already available via `GH |` or `GC |` (free-before-paid strategy).

**Minimum IAM policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/*"
    }
  ]
}
```

**Steps:**

1. Go to [IAM Users](https://console.aws.amazon.com/iam/home#/users)
2. Create or select a user and attach the policy above
3. Under **Security credentials**, create an access key (choose "Application running outside AWS")
4. Copy the Access Key ID and Secret Access Key into your `.env`
5. Enable the models you want at [Bedrock Model Access](https://console.aws.amazon.com/bedrock/home#/modelaccess)

**Recommended regions** (broadest model selection):

| Region | Location |
|---|---|
| `us-east-1` | US East (N. Virginia) — widest selection |
| `us-west-2` | US West (Oregon) |
| `eu-west-1` | Europe (Ireland) |
| `ap-southeast-1` | Asia Pacific (Singapore) |

---

## Available Models

### `GH |` — GitHub Models (PAT-backed, free quota)

Discovered dynamically at startup from `https://models.inference.ai.azure.com/models`. Typical models include:

| Chat | Embeddings |
|---|---|
| `GH \| gpt-4o`, `GH \| gpt-4.1`, `GH \| gpt-4.1-mini` | `GH \| Cohere-embed-v3-english` |
| `GH \| Meta-Llama-3.1-405B-Instruct`, `GH \| Meta-Llama-3.1-8B-Instruct` | `GH \| Cohere-embed-v3-multilingual` |
| `GH \| Mistral-Nemo`, `GH \| Mistral-Large` | `GH \| text-embedding-3-large` |
| *(and more, depending on your account)* | `GH \| text-embedding-3-small` |

To see exactly what your token can access:

```bash
curl -H "Authorization: Bearer $GITHUB_TOKEN" https://models.inference.ai.azure.com/models
```

### `GC |` — GitHub Copilot Models (OAuth, requires Copilot subscription)

Discovered dynamically after login. Includes Claude (Haiku/Sonnet/Opus), GPT-4o, GPT-4.1, Gemini, Grok, Llama, and more — exact availability depends on your Copilot plan. Models that are already available under `GH |` are deduplicated (free quota is always preferred).

### `BR |` — AWS Bedrock Models (pay-per-use)

Includes Claude 3/3.5/4 (Haiku, Sonnet, Opus), Mistral (Large, Small, Mixtral), Llama 3.1/3.2/3.3/4, Amazon Nova, DeepSeek R1, Palmyra, and Titan Embed models. Only models **not** already covered by `GH |` or `GC |` are advertised.

### `155 |` — Remote Ollama Models

Any model installed on the Ollama server at `REMOTE_OLLAMA_URL` is automatically listed with the `155 |` prefix, e.g. `155 | llama3.2`, `155 | mistral`.

---

## API Endpoints

### OpenAI-compatible (recommended for Continue.dev, Cursor, etc.)

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List all models in OpenAI format |
| `POST` | `/v1/chat/completions` | Chat completions — streaming + non-streaming, all model types |
| `POST` | `/v1/embeddings` | Embeddings — `GH \|`, `GC \|`, `BR \|`, or remote Ollama |

### Ollama-native

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/tags` | List all models in Ollama format |
| `POST` | `/api/show` | Model metadata |
| `POST` | `/api/chat` | Ollama-style chat (NDJSON streaming or blocking) |
| `POST` | `/api/generate` | Ollama-style generate (NDJSON streaming or blocking) |
| `POST` | `/api/chat/completions` | OpenAI-compat chat (via Ollama path) |
| `POST` | `/api/embeddings` | Embeddings (legacy alias) |
| `POST` | `/api/embed` | Embeddings |
| `POST` | `/api/pull` | Pull a model on the remote Ollama server |
| `GET` | `/api/ps` | List running models on the remote server |
| `DELETE` | `/api/delete` | Delete a model on the remote server |
| `POST` | `/api/copy` | Copy a model on the remote server |

### Auth & Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/auth/copilot` | Start Copilot browser login (device flow) |
| `GET` | `/auth/status` | Check Copilot login state and `GC \|` model count |
| `GET` | `/health` | Proxy health — remote Ollama, GitHub, Copilot, Bedrock status |
| `GET` | `/` | Index — version, model counts, endpoint list |

---

## Continue.dev Configuration

Use the **OpenAI provider** with the proxy's `/v1/` base URL. The `apiKey` value is ignored by the proxy but must be non-empty.

```json
{
  "models": [
    {
      "title": "GH | gpt-4.1",
      "provider": "openai",
      "model": "GH | gpt-4.1",
      "apiBase": "http://127.0.0.1:5000/v1/",
      "apiKey": "proxy"
    },
    {
      "title": "GC | claude-sonnet-4",
      "provider": "openai",
      "model": "GC | claude-sonnet-4",
      "apiBase": "http://127.0.0.1:5000/v1/",
      "apiKey": "proxy"
    },
    {
      "title": "BR | us.anthropic.claude-3-5-haiku-20241022-v1:0",
      "provider": "openai",
      "model": "BR | us.anthropic.claude-3-5-haiku-20241022-v1:0",
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
  ],
  "embeddingsProvider": {
    "provider": "openai",
    "model": "GH | text-embedding-3-small",
    "apiBase": "http://127.0.0.1:5000/v1/",
    "apiKey": "proxy"
  }
}
```

---

## How It Works

1. **Startup** — the proxy probes `GH |` models against the live GitHub Models API, checks for a saved Copilot token to discover `GC |` models, and builds the `BR |` catalogue from a static list filtered to exclude anything already available for free.

2. **`GET /v1/models` or `GET /api/tags`** — returns the combined catalogue: remote Ollama models + all `GH |`, `GC |`, and `BR |` entries.

3. **Request routing:**
   - `155 | <model>` → strips prefix and forwards verbatim to `REMOTE_OLLAMA_URL`
   - `GH | <model>` / `GC | <model>` → calls the Azure AI Inference SDK against the appropriate endpoint with the correct bearer token
   - `BR | <model>` → calls the AWS Bedrock `converse` / `converse_stream` / `invoke_model` API via boto3
   - Bare or unknown model names → forwarded to the remote Ollama server as-is

4. **Format translation** — all responses are translated into Ollama NDJSON format for `/api/chat` and `/api/generate`, or into OpenAI SSE / JSON format for `/v1/chat/completions`. Both streaming and non-streaming are fully supported across all model types.

---

## Troubleshooting

### `GITHUB_TOKEN is not configured`
Add `GITHUB_TOKEN=github_pat_...` to your `.env` and restart.

### `GH |` models not appearing at startup
The proxy probes each candidate model. If your token has no `models:read` access or GitHub Models is rate-limiting you, probes will fail silently and a fallback catalogue will be used. Check the startup logs for probe results.

### `GC |` models not appearing
You need to complete the Copilot device-flow login. Open [http://localhost:5000/auth/copilot](http://localhost:5000/auth/copilot) and follow the instructions.

### `BR |` models not appearing
- Check that `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set in `.env`
- Verify the IAM user has the Bedrock invoke policy attached
- Make sure you have enabled the specific models in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) for your region
- `boto3` must be installed: `pip install boto3`

### `self signed certificate in certificate chain`
You are behind a corporate proxy that intercepts TLS. Set `DISABLE_SSL_VERIFY=true` in `.env` as a temporary workaround, or point `REQUESTS_CA_BUNDLE` at your corporate root certificate for a proper fix.

### Remote Ollama models (`155 |`) not showing
Verify the remote server is reachable:
```bash
curl http://192.168.1.155:11434/api/tags
```
Update `REMOTE_OLLAMA_URL` in `.env` if your server is at a different address.

### Port already in use
Set `PORT=<number>` in your `.env`.

### Model name not found / 404
Model names are case-sensitive and include the prefix and spaces, e.g. `GH | gpt-4o`. The proxy normalises dots to dashes automatically (e.g. `claude-haiku-4.5` → `GH | claude-haiku-4-5`), but the prefix and spacing must match. Use `GET /v1/models` or `GET /api/tags` to see the exact names.

---

## License

MIT