"""
Ollama Proxy Server with GitHub Models + GitHub Copilot support.

- Ollama API requests are proxied to the remote server at REMOTE_OLLAMA_URL.
- GH | models use the azure-ai-inference SDK against
  https://models.inference.ai.azure.com with a standard GitHub PAT.
- GC | models use the GitHub Copilot Chat API (api.githubcopilot.com) via
  an OAuth token obtained through the GitHub device flow. Run:
    GET http://localhost:5000/auth/copilot
  and follow the instructions to log in through your browser.
- Responses are translated into Ollama-format NDJSON so any Ollama client
  (Continue.dev, Open WebUI, etc.) works transparently.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Silence the very noisy azure-core HTTP logger unless you want it
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REMOTE_OLLAMA_URL = os.getenv("REMOTE_OLLAMA_URL", "http://192.168.1.155:11434")
PORT = int(os.getenv("PORT", "5000"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

# ---------------------------------------------------------------------------
# GitHub Copilot OAuth — device flow
# ---------------------------------------------------------------------------

# The public client_id used by the VS Code GitHub Copilot extension.
COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_CHAT_ENDPOINT = "https://api.githubcopilot.com"
COPILOT_CHAT_HEADERS = {
    "Copilot-Integration-Id": "vscode-chat",
    "Editor-Version": "vscode/1.97.0",
    "Content-Type": "application/json",
}

# In-memory Copilot OAuth token (populated by device flow or from .env).
# Access via _get_copilot_token() / _set_copilot_token().
_copilot_token: str = os.getenv("GITHUB_TOKEN_COPILOT", "")
_copilot_token_lock = threading.Lock()

# Tracks a pending device-flow authorisation.
_pending_device_flow: dict = {}


def _get_copilot_token() -> str:
    with _copilot_token_lock:
        return _copilot_token


def _set_copilot_token(token: str) -> None:
    global _copilot_token
    with _copilot_token_lock:
        _copilot_token = token
    # Persist to .env so it survives restarts
    env_path = Path(".env")
    try:
        if env_path.exists():
            text = env_path.read_text()
            if "GITHUB_TOKEN_COPILOT=" in text:
                lines = [
                    f"GITHUB_TOKEN_COPILOT={token}\n"
                    if l.startswith("GITHUB_TOKEN_COPILOT=")
                    else l
                    for l in text.splitlines(keepends=True)
                ]
                env_path.write_text("".join(lines))
            else:
                with env_path.open("a") as f:
                    f.write(f"\nGITHUB_TOKEN_COPILOT={token}\n")
        else:
            env_path.write_text(f"GITHUB_TOKEN_COPILOT={token}\n")
        log.info("Copilot token persisted to .env")
    except Exception as e:
        log.warning("Could not persist Copilot token to .env: %s", e)


def _discover_copilot_models() -> list[dict]:
    """Query api.githubcopilot.com/models and return catalogue entries."""
    token = _get_copilot_token()
    if not token:
        return []
    try:
        resp = requests.get(
            f"{COPILOT_CHAT_ENDPOINT}/models",
            headers={**COPILOT_CHAT_HEADERS, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
        # Only keep chat models (skip embedding / image models)
        SKIP = {"embeddings", "embed", "image", "dall-e", "whisper", "tts"}
        entries = []
        for m in raw:
            mid = m.get("id", "")
            if not mid:
                continue
            if any(kw in mid.lower() for kw in SKIP):
                continue
            entries.append(
                {
                    "name": f"{GC_PREFIX}{mid}",
                    "sdk_name": mid,
                    "token": "copilot",
                    "size": 0,
                    "modified_at": "2024-01-01T00:00:00Z",
                    "details": {
                        "family": mid.split("-")[0].lower(),
                        "parameter_size": "unknown",
                    },
                }
            )
        log.info("Discovered %d Copilot models (GC |)", len(entries))
        return entries
    except Exception as e:
        log.warning("Copilot model discovery failed: %s", e)
        return []


def _rebuild_catalogue() -> None:
    """Rebuild GITHUB_MODELS + GITHUB_MODEL_MAP after a new Copilot token arrives."""
    global GITHUB_MODELS, GITHUB_MODEL_MAP
    gh_models = _discover_models(GITHUB_TOKEN, GH_PREFIX, "standard")
    gc_models = _discover_copilot_models()
    # Avoid duplicating models that appear in both endpoints
    gc_sdk_names = {m["sdk_name"] for m in gh_models}
    unique_gc = [m for m in gc_models if m["sdk_name"] not in gc_sdk_names]
    GITHUB_MODELS = gh_models + unique_gc
    GITHUB_MODEL_MAP = {m["name"]: m for m in GITHUB_MODELS}
    log.info(
        "Catalogue rebuilt: %d GH | + %d GC | = %d total",
        len(gh_models),
        len(unique_gc),
        len(GITHUB_MODELS),
    )


# ---------------------------------------------------------------------------
# Model discovery — standard GitHub Models (PAT-based)
# ---------------------------------------------------------------------------

_FALLBACK_MODELS = [
    {"sdk_name": "gpt-4o", "family": "gpt", "size": "unknown"},
    {"sdk_name": "gpt-4o-mini", "family": "gpt", "size": "unknown"},
    {"sdk_name": "gpt-4.1", "family": "gpt", "size": "unknown"},
    {"sdk_name": "gpt-4.1-mini", "family": "gpt", "size": "unknown"},
    {"sdk_name": "Meta-Llama-3.1-405B-Instruct", "family": "llama", "size": "405B"},
    {"sdk_name": "Meta-Llama-3.1-8B-Instruct", "family": "llama", "size": "8B"},
]


def _discover_models(token: str, prefix: str, token_label: str) -> list[dict]:
    """
    Query the GitHub Models catalogue endpoint and return a list of proxy
    model entries. Falls back to the hardcoded list on any error.
    """
    if not token:
        return []

    fallback = _FALLBACK_MODELS

    try:
        resp = requests.get(
            f"{GITHUB_INFERENCE_ENDPOINT}/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        raw_models = resp.json()

        # Filter to chat-capable models only (skip embedding models)
        EMBEDDING_KEYWORDS = {"embed", "embedding"}
        chat_models = [
            m
            for m in raw_models
            if not any(kw in m.get("name", "").lower() for kw in EMBEDDING_KEYWORDS)
        ]

        entries = []
        for m in chat_models:
            sdk_name = m.get("name", "")
            if not sdk_name:
                continue
            entries.append(
                {
                    "name": f"{prefix}{sdk_name}",
                    "sdk_name": sdk_name,
                    "token": token_label,
                    "size": 0,
                    "modified_at": "2024-01-01T00:00:00Z",
                    "details": {
                        "family": sdk_name.split("-")[0].lower(),
                        "parameter_size": "unknown",
                    },
                }
            )

        log.info(
            "Discovered %d chat models for %s token (prefix '%s')",
            len(entries),
            token_label,
            prefix,
        )
        return entries

    except Exception as e:
        log.warning(
            "Model discovery failed for %s token (%s) — using fallback catalogue",
            token_label,
            e,
        )
        return [
            {
                "name": f"{prefix}{f['sdk_name']}",
                "sdk_name": f["sdk_name"],
                "token": token_label,
                "size": 0,
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"family": f["family"], "parameter_size": f["size"]},
            }
            for f in fallback
        ]


# ---------------------------------------------------------------------------
# GitHub model catalogue — built at startup via live API discovery
# ---------------------------------------------------------------------------

GH_PREFIX = "GH | "
GC_PREFIX = "GC | "
REMOTE_PREFIX = "155 | "

# Populated at module load time; rebuilt after Copilot login via _rebuild_catalogue().
GITHUB_MODELS: list[dict] = _discover_models(GITHUB_TOKEN, GH_PREFIX, "standard")
GITHUB_MODEL_MAP: dict[str, dict] = {m["name"]: m for m in GITHUB_MODELS}

# Kick off Copilot discovery now if a token is already in .env
if _get_copilot_token():
    _rebuild_catalogue()


def _active_github_models() -> list[dict]:
    """Return all currently known models."""
    return GITHUB_MODELS


def _token_for_model(name: str) -> str:
    """Return the correct bearer token for the given canonical model name."""
    entry = GITHUB_MODEL_MAP.get(name, {})
    if entry.get("token") == "copilot":
        token = _get_copilot_token()
        if not token:
            raise RuntimeError(
                f"'{name}' is a Copilot model. "
                "Log in first: GET http://localhost:5000/auth/copilot"
            )
        return token
    if not GITHUB_TOKEN:
        raise RuntimeError(f"'{name}' requires GITHUB_TOKEN, which is not set.")
    return GITHUB_TOKEN


def is_github_model(name: str) -> bool:
    """Accept prefixed, bare, and dot-notation names (e.g. 'claude-haiku-4.5')."""
    return _canonical_gh_name(name) in GITHUB_MODEL_MAP


def _canonical_gh_name(name: str) -> str:
    """Return the prefixed display name regardless of how the client sent it.

    Handles:
      - "GH | gpt-4o"            (already canonical, standard)
      - "GC | claude-haiku-4.5"  (already canonical, copilot)
      - "gpt-4o"                 (bare)
      - "claude-haiku-4.5"       (bare with dots — tried with both prefixes)
    """
    if name in GITHUB_MODEL_MAP:
        return name
    for prefix in (GH_PREFIX, GC_PREFIX):
        prefixed = prefix + name
        if prefixed in GITHUB_MODEL_MAP:
            return prefixed
    # Normalise dots to dashes and retry with both prefixes
    normalised = name.replace(".", "-")
    if normalised in GITHUB_MODEL_MAP:
        return normalised
    for prefix in (GH_PREFIX, GC_PREFIX):
        prefixed_normalised = prefix + normalised
        if prefixed_normalised in GITHUB_MODEL_MAP:
            return prefixed_normalised
    return name


def get_sdk_name(name: str) -> str:
    """Return the GitHub inference endpoint model ID for a display name."""
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(name))
    return entry["sdk_name"] if entry else name


# ---------------------------------------------------------------------------
# Global Flask error handlers — always JSON, never HTML
# ---------------------------------------------------------------------------


@app.errorhandler(404)
def not_found(e):
    log.warning("404 %s %s", request.method, request.path)
    return jsonify({"error": "not found", "path": request.path}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    log.warning("405 %s %s", request.method, request.path)
    return jsonify({"error": "method not allowed"}), 405


@app.errorhandler(Exception)
def unhandled(e):
    log.exception("Unhandled exception: %s", e)
    return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def proxy_request(method: str, path: str, data=None, params=None, stream: bool = False):
    """Forward a request to the remote Ollama server."""
    url = f"{REMOTE_OLLAMA_URL}{path}"
    log.debug("-> %s %s (stream=%s)", method, url, stream)
    resp = requests.request(
        method,
        url,
        json=data,
        params=params,
        headers={"Content-Type": "application/json"},
        timeout=300,
        stream=stream,
    )
    log.debug("<- %s %s => %s", method, url, resp.status_code)
    return resp


def stream_chunks(upstream_response):
    """Yield raw bytes from an upstream streaming requests.Response."""
    for chunk in upstream_response.iter_content(chunk_size=None):
        if chunk:
            yield chunk


def make_proxy_response(
    resp, default_content_type: str = "application/json"
) -> Response:
    return Response(
        resp.content,
        status=resp.status_code,
        content_type=resp.headers.get("Content-Type", default_content_type),
    )


def make_streaming_proxy_response(
    resp, default_content_type: str = "application/x-ndjson"
) -> Response:
    return Response(
        stream_with_context(stream_chunks(resp)),
        status=resp.status_code,
        content_type=resp.headers.get("Content-Type", default_content_type),
    )


# ---------------------------------------------------------------------------
# GitHub SDK helpers
# ---------------------------------------------------------------------------


def _build_github_client(model_name: str) -> ChatCompletionsClient:
    """Build a ChatCompletionsClient using the correct endpoint and token."""
    entry = GITHUB_MODEL_MAP.get(model_name, {})
    token = _token_for_model(model_name)
    endpoint = (
        COPILOT_CHAT_ENDPOINT
        if entry.get("token") == "copilot"
        else GITHUB_INFERENCE_ENDPOINT
    )
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )
    if entry.get("token") == "copilot":
        # The Copilot endpoint requires extra headers
        client._config.headers_policy.headers.update(COPILOT_CHAT_HEADERS)
    return client


def _ollama_messages_to_sdk(messages: list[dict]):
    """Convert Ollama-style message dicts to azure-ai-inference message objects."""
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            result.append(SystemMessage(content))
        elif role == "assistant":
            result.append(AssistantMessage(content))
        else:
            result.append(UserMessage(content))
    return result


def _github_chat_streaming(model_name: str, messages: list[dict]):
    """
    Call GitHub Models via the SDK with stream=True and yield Ollama-format
    NDJSON lines. Each delta becomes one line; a final done=True line closes
    the stream.
    """
    sdk_name = get_sdk_name(model_name)
    sdk_messages = _ollama_messages_to_sdk(messages)
    client = _build_github_client(model_name)

    try:
        response = client.complete(
            model=sdk_name,
            messages=sdk_messages,
            stream=True,
        )

        for update in response:
            if not update.choices:
                continue
            delta = update.choices[0].delta
            content = delta.content if delta and delta.content else ""
            finish_reason = update.choices[0].finish_reason

            is_done = finish_reason is not None

            chunk = {
                "model": model_name,
                "created_at": now_iso(),
                "message": {"role": "assistant", "content": content},
                "done": is_done,
            }
            if is_done:
                chunk["done_reason"] = str(finish_reason) if finish_reason else "stop"
                chunk["total_duration"] = 0
                chunk["load_duration"] = 0
                chunk["prompt_eval_count"] = 0
                chunk["eval_count"] = 0

            yield json.dumps(chunk) + "\n"

    except HttpResponseError as e:
        log.error("GitHub SDK error: %s %s", e.status_code, e.message)
        error_chunk = {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "error",
            "error": f"{e.status_code}: {e.message}",
        }
        yield json.dumps(error_chunk) + "\n"
    finally:
        client.close()


def _github_chat_blocking(model_name: str, messages: list[dict]) -> dict:
    """
    Call GitHub Models via the SDK with stream=False and return a single
    Ollama-format response dict.
    """
    sdk_name = get_sdk_name(model_name)
    sdk_messages = _ollama_messages_to_sdk(messages)
    client = _build_github_client(model_name)

    try:
        response = client.complete(
            model=sdk_name,
            messages=sdk_messages,
            stream=False,
        )
        content = response.choices[0].message.content if response.choices else ""
        usage = response.usage

        return {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": usage.prompt_tokens if usage else 0,
            "eval_count": usage.completion_tokens if usage else 0,
        }

    except HttpResponseError as e:
        log.error("GitHub SDK error: %s %s", e.status_code, e.message)
        return {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "error",
            "error": f"{e.status_code}: {e.message}",
        }
    finally:
        client.close()


def _github_generate_blocking(model_name: str, prompt: str, system: str = "") -> dict:
    """
    Wrap a plain /api/generate prompt as a chat call to GitHub Models
    and return an Ollama /api/generate-format response dict.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    chat_result = _github_chat_blocking(model_name, messages)

    return {
        "model": model_name,
        "created_at": chat_result["created_at"],
        "response": chat_result["message"]["content"],
        "done": True,
        "done_reason": chat_result.get("done_reason", "stop"),
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": chat_result.get("prompt_eval_count", 0),
        "eval_count": chat_result.get("eval_count", 0),
    }


def _github_generate_streaming(model_name: str, prompt: str, system: str = ""):
    """
    Wrap a plain /api/generate prompt as a streaming chat call and yield
    Ollama /api/generate-format NDJSON lines.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for chat_line in _github_chat_streaming(model_name, messages):
        chat_chunk = json.loads(chat_line)
        gen_chunk = {
            "model": model_name,
            "created_at": chat_chunk["created_at"],
            "response": chat_chunk.get("message", {}).get("content", ""),
            "done": chat_chunk["done"],
        }
        if chat_chunk["done"]:
            gen_chunk["done_reason"] = chat_chunk.get("done_reason", "stop")
            gen_chunk["total_duration"] = 0
            gen_chunk["load_duration"] = 0
            gen_chunk["prompt_eval_count"] = chat_chunk.get("prompt_eval_count", 0)
            gen_chunk["eval_count"] = chat_chunk.get("eval_count", 0)
        yield json.dumps(gen_chunk) + "\n"


# ---------------------------------------------------------------------------
# /api/tags
# ---------------------------------------------------------------------------


@app.route("/api/tags", methods=["GET"])
def get_tags():
    """List all models — remote Ollama models plus GitHub catalogue."""
    try:
        resp = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        remote_models = resp.json().get("models", [])
        log.debug("Fetched %d remote models", len(remote_models))
    except Exception as e:
        log.warning("Could not reach remote Ollama: %s", e)
        remote_models = []

    # Prefix each remote Ollama model name with "155 | "
    for m in remote_models:
        if not m.get("name", "").startswith(REMOTE_PREFIX):
            m["name"] = REMOTE_PREFIX + m["name"]

    # Only show models whose token is configured; strip internal fields
    _INTERNAL_KEYS = {"sdk_name", "token"}
    github_entries = [
        {k: v for k, v in m.items() if k not in _INTERNAL_KEYS}
        for m in _active_github_models()
    ]
    return jsonify({"models": remote_models + github_entries})


# ---------------------------------------------------------------------------
# /api/show
# ---------------------------------------------------------------------------


@app.route("/api/show", methods=["POST"])
def show():
    """Return model metadata. GitHub models get a synthetic response."""
    data = request.json or {}
    model_name = data.get("model") or data.get("name", "")
    log.debug("/api/show model=%r", model_name)

    if is_github_model(model_name):
        entry = GITHUB_MODEL_MAP[_canonical_gh_name(model_name)]
        return jsonify(
            {
                "license": "",
                "modelfile": f"# GitHub Models via azure-ai-inference: {model_name}",
                "parameters": "",
                "template": "{{ .Prompt }}",
                "details": {
                    "format": "api",
                    "family": entry["details"].get("family", "github"),
                    "families": [entry["details"].get("family", "github")],
                    "parameter_size": entry["details"].get("parameter_size", "unknown"),
                    "quantization_level": "none",
                },
                "model_info": {},
            }
        )

    resp = proxy_request("POST", "/api/show", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/chat  (Ollama-native NDJSON chat)
# ---------------------------------------------------------------------------


@app.route("/api/chat", methods=["POST"])
def chat():
    """Ollama /api/chat — real GitHub SDK call or proxy to remote Ollama."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", True)
    messages = data.get("messages", [])
    log.debug(
        "/api/chat model=%r stream=%s msgs=%d", model_name, do_stream, len(messages)
    )

    # Strip "155 | " prefix and forward to remote Ollama
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
        resp = proxy_request("POST", "/api/chat", data=data, stream=do_stream)
        if do_stream:
            return make_streaming_proxy_response(resp, "application/x-ndjson")
        return make_proxy_response(resp)

    if is_github_model(model_name):
        if not GITHUB_TOKEN:
            return jsonify(
                {"error": "GITHUB_TOKEN is not configured on the proxy server."}
            ), 500

        if do_stream:
            return Response(
                stream_with_context(_github_chat_streaming(model_name, messages)),
                status=200,
                content_type="application/x-ndjson",
            )
        else:
            result = _github_chat_blocking(model_name, messages)
            if "error" in result:
                return jsonify(result), 502
            return jsonify(result)

    resp = proxy_request("POST", "/api/chat", data=data, stream=do_stream)
    if do_stream:
        return make_streaming_proxy_response(resp, "application/x-ndjson")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/generate  (Ollama-native NDJSON generate)
# ---------------------------------------------------------------------------


@app.route("/api/generate", methods=["POST"])
def generate():
    """Ollama /api/generate — real GitHub SDK call or proxy to remote Ollama."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", True)
    prompt = data.get("prompt", "")
    system = data.get("system", "")
    log.debug("/api/generate model=%r stream=%s", model_name, do_stream)

    # Strip "155 | " prefix and forward to remote Ollama
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
        resp = proxy_request("POST", "/api/generate", data=data, stream=do_stream)
        if do_stream:
            return make_streaming_proxy_response(resp, "application/x-ndjson")
        return make_proxy_response(resp)

    if is_github_model(model_name):
        model_name = _canonical_gh_name(model_name)
        if not GITHUB_TOKEN:
            return jsonify(
                {"error": "GITHUB_TOKEN is not configured on the proxy server."}
            ), 500

        if do_stream:
            return Response(
                stream_with_context(
                    _github_generate_streaming(model_name, prompt, system)
                ),
                status=200,
                content_type="application/x-ndjson",
            )
        else:
            result = _github_generate_blocking(model_name, prompt, system)
            if "error" in result:
                return jsonify(result), 502
            return jsonify(result)

    resp = proxy_request("POST", "/api/generate", data=data, stream=do_stream)
    if do_stream:
        return make_streaming_proxy_response(resp, "application/x-ndjson")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/chat/completions  (OpenAI-compat pass-through)
# ---------------------------------------------------------------------------


@app.route("/api/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions — proxy to remote Ollama."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", False)
    log.debug("/api/chat/completions model=%r stream=%s", model_name, do_stream)

    # Strip "155 | " prefix and forward to remote Ollama
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
        resp = proxy_request(
            "POST", "/api/chat/completions", data=data, stream=do_stream
        )
        if do_stream:
            return make_streaming_proxy_response(resp, "text/event-stream")
        return make_proxy_response(resp)

    # For GitHub models, re-use the Ollama-format chat handler and wrap the
    # result in an OpenAI-compat envelope (non-streaming only for simplicity).
    if is_github_model(model_name):
        model_name = _canonical_gh_name(model_name)
        if not GITHUB_TOKEN:
            return jsonify(
                {
                    "error": {
                        "message": "GITHUB_TOKEN is not configured.",
                        "type": "api_error",
                    }
                }
            ), 500

        messages = data.get("messages", [])
        result = _github_chat_blocking(model_name, messages)
        if "error" in result:
            return jsonify(result), 502

        return jsonify(
            {
                "id": "chatcmpl-github",
                "object": "chat.completion",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": result["message"],
                        "finish_reason": result.get("done_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": (
                        result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    ),
                },
            }
        )

    resp = proxy_request("POST", "/api/chat/completions", data=data, stream=do_stream)
    if do_stream:
        return make_streaming_proxy_response(resp, "text/event-stream")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/embeddings  and  /api/embed
# ---------------------------------------------------------------------------


@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    data = request.json or {}
    log.debug("/api/embeddings model=%r", data.get("model"))
    resp = proxy_request("POST", "/api/embeddings", data=data)
    return make_proxy_response(resp)


@app.route("/api/embed", methods=["POST"])
def embed():
    data = request.json or {}
    log.debug("/api/embed model=%r", data.get("model"))
    resp = proxy_request("POST", "/api/embed", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/pull
# ---------------------------------------------------------------------------


@app.route("/api/pull", methods=["POST"])
def pull():
    data = request.json or {}
    do_stream = data.get("stream", True)
    log.debug("/api/pull model=%r stream=%s", data.get("model"), do_stream)
    resp = proxy_request("POST", "/api/pull", data=data, stream=do_stream)
    if do_stream:
        return make_streaming_proxy_response(resp, "application/x-ndjson")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/ps
# ---------------------------------------------------------------------------


@app.route("/api/ps", methods=["GET"])
def ps():
    resp = proxy_request("GET", "/api/ps")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/delete
# ---------------------------------------------------------------------------


@app.route("/api/delete", methods=["DELETE"])
def delete():
    data = request.json or {}
    resp = proxy_request("DELETE", "/api/delete", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/copy
# ---------------------------------------------------------------------------


@app.route("/api/copy", methods=["POST"])
def copy():
    data = request.json or {}
    resp = proxy_request("POST", "/api/copy", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /health  and  /
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /auth/copilot  — GitHub device flow login
# /auth/status   — check Copilot login state
# ---------------------------------------------------------------------------


@app.route("/auth/copilot", methods=["GET"])
def auth_copilot():
    """Kick off the GitHub device flow to obtain a Copilot OAuth token."""
    global _pending_device_flow

    if _get_copilot_token():
        # Already logged in — check it still works
        try:
            resp = requests.get(
                f"{COPILOT_CHAT_ENDPOINT}/models",
                headers={
                    **COPILOT_CHAT_HEADERS,
                    "Authorization": f"Bearer {_get_copilot_token()}",
                },
                timeout=5,
            )
            if resp.status_code == 200:
                model_count = len(resp.json().get("data", []))
                return jsonify(
                    {
                        "status": "already_authenticated",
                        "message": "Copilot token is valid.",
                        "copilot_models_available": model_count,
                    }
                )
        except Exception:
            pass
        # Token is stale — fall through to re-auth
        log.info("Copilot token stale — starting new device flow")

    # Request device code
    r = requests.post(
        "https://github.com/login/device/code",
        headers={"Accept": "application/json"},
        data={"client_id": COPILOT_CLIENT_ID, "scope": ""},
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()

    _pending_device_flow = {
        "device_code": d["device_code"],
        "interval": d.get("interval", 5),
        "expires_at": time.time() + d.get("expires_in", 900),
    }

    # Poll in a background thread so the endpoint returns immediately
    threading.Thread(target=_poll_device_flow, daemon=True).start()

    return jsonify(
        {
            "status": "pending",
            "message": (
                f"1. Open https://github.com/login/device in your browser\n"
                f"2. Enter code: {d['user_code']}\n"
                f"3. Approve the request\n"
                f"4. Poll GET /auth/status until status is 'authenticated'"
            ),
            "verification_uri": d["verification_uri"],
            "user_code": d["user_code"],
            "expires_in_seconds": d.get("expires_in", 900),
        }
    )


def _poll_device_flow() -> None:
    """Background thread: poll GitHub until the user approves or the code expires."""
    global _pending_device_flow
    flow = _pending_device_flow
    interval = flow["interval"]

    log.info("Device flow polling started")
    while time.time() < flow["expires_at"]:
        time.sleep(interval)
        try:
            r = requests.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": COPILOT_CLIENT_ID,
                    "device_code": flow["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                timeout=10,
            )
            data = r.json()
        except Exception as e:
            log.warning("Device flow poll error: %s", e)
            continue

        if "access_token" in data:
            token = data["access_token"]
            _set_copilot_token(token)
            _pending_device_flow = {}
            log.info("Copilot OAuth token obtained — rebuilding model catalogue")
            _rebuild_catalogue()
            return
        elif data.get("error") == "slow_down":
            interval += 5
        elif data.get("error") == "authorization_pending":
            pass
        else:
            log.warning("Device flow ended: %s", data.get("error", data))
            _pending_device_flow = {}
            return

    log.warning("Device flow expired without approval")
    _pending_device_flow = {}


@app.route("/auth/status", methods=["GET"])
def auth_status():
    """Report the current Copilot authentication state."""
    token = _get_copilot_token()
    pending = bool(_pending_device_flow)

    if pending:
        return jsonify(
            {
                "status": "pending",
                "message": "Waiting for browser approval. Check /auth/copilot for the code.",
            }
        )

    if not token:
        return jsonify(
            {
                "status": "unauthenticated",
                "message": "No Copilot token. Call GET /auth/copilot to log in.",
                "gc_models": 0,
            }
        )

    gc_count = sum(1 for m in GITHUB_MODELS if m.get("token") == "copilot")
    return jsonify(
        {
            "status": "authenticated",
            "message": f"{gc_count} GC | Copilot models available.",
            "gc_models": gc_count,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    try:
        resp = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        remote_status = "ok" if resp.status_code == 200 else "error"
    except Exception:
        remote_status = "error"

    gh_status = "ready" if GITHUB_TOKEN else "not set — add GITHUB_TOKEN to .env"
    gc_token = _get_copilot_token()
    gc_count = sum(1 for m in GITHUB_MODELS if m.get("token") == "copilot")
    gc_status = (
        f"authenticated ({gc_count} models)"
        if gc_token
        else "not logged in — GET /auth/copilot"
    )
    status = "healthy" if remote_status == "ok" else "degraded"

    return jsonify(
        {
            "status": status,
            "remote_ollama": remote_status,
            "github_models": gh_status,
            "copilot_models": gc_status,
        }
    )


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "message": "Ollama Proxy with GitHub Models",
            "version": "1.5.0",
            "endpoints": [
                "GET    /api/tags              — list all models",
                "POST   /api/show              — model metadata",
                "POST   /api/chat              — Ollama-style chat (NDJSON)",
                "POST   /api/generate          — Ollama-style generate (NDJSON)",
                "POST   /api/chat/completions  — OpenAI-compat chat completions",
                "POST   /api/embeddings        — embeddings (legacy alias)",
                "POST   /api/embed             — embeddings",
                "POST   /api/pull              — pull a model",
                "GET    /api/ps                — running models",
                "DELETE /api/delete            — delete a model",
                "POST   /api/copy              — copy a model",
                "GET    /v1/models             — OpenAI-compat model list",
                "POST   /v1/chat/completions   — OpenAI-compat chat completions",
                "GET    /auth/copilot          — start Copilot browser login (device flow)",
                "GET    /auth/status           — check Copilot login state",
                "GET    /health                — health check",
            ],
        }
    )


# ---------------------------------------------------------------------------
# /v1/models  and  /v1/chat/completions  (OpenAI-compat surface for Continue.dev)
# ---------------------------------------------------------------------------


def _all_models_as_openai() -> list[dict]:
    """Return every model (remote Ollama + GitHub) in OpenAI /v1/models format."""
    models = []

    try:
        resp = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        for m in resp.json().get("models", []):
            raw_name = m.get("name", "")
            display = (
                raw_name
                if raw_name.startswith(REMOTE_PREFIX)
                else REMOTE_PREFIX + raw_name
            )
            models.append(
                {
                    "id": display,
                    "object": "model",
                    "created": 0,
                    "owned_by": "ollama",
                }
            )
    except Exception as e:
        log.warning("Could not reach remote Ollama for /v1/models: %s", e)

    for m in _active_github_models():
        models.append(
            {
                "id": m["name"],
                "object": "model",
                "created": 0,
                "owned_by": "github",
            }
        )

    return models


@app.route("/v1/models", methods=["GET"])
def v1_models():
    """OpenAI-compat GET /v1/models — lists all proxied models."""
    return jsonify({"object": "list", "data": _all_models_as_openai()})


def _github_chat_streaming_openai(model_name: str, messages: list[dict]):
    """
    Stream GitHub model response as OpenAI-compat SSE chunks
    (data: {...}\\n\\n lines, terminated with data: [DONE]).
    """
    sdk_name = get_sdk_name(model_name)
    sdk_messages = _ollama_messages_to_sdk(messages)
    client = _build_github_client(model_name)
    created = int(datetime.now(timezone.utc).timestamp())

    try:
        response = client.complete(
            model=sdk_name,
            messages=sdk_messages,
            stream=True,
        )

        for update in response:
            if not update.choices:
                continue
            delta = update.choices[0].delta
            content = delta.content if delta and delta.content else ""
            finish_reason = update.choices[0].finish_reason

            chunk = {
                "id": "chatcmpl-github",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": str(finish_reason) if finish_reason else None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    except HttpResponseError as e:
        log.error("GitHub SDK streaming error: %s %s", e.status_code, e.message)
        err_chunk = {
            "error": {"message": f"{e.status_code}: {e.message}", "type": "api_error"}
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        client.close()


@app.route("/v1/chat/completions", methods=["POST"])
def v1_chat_completions():
    """OpenAI-compat POST /v1/chat/completions — routes to GitHub or remote Ollama."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", False)
    messages = data.get("messages", [])
    log.debug(
        "/v1/chat/completions model=%r stream=%s msgs=%d",
        model_name,
        do_stream,
        len(messages),
    )

    # Strip "155 | " prefix and forward to remote Ollama's OpenAI-compat endpoint
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
        resp = proxy_request(
            "POST", "/v1/chat/completions", data=data, stream=do_stream
        )
        if do_stream:
            return make_streaming_proxy_response(resp, "text/event-stream")
        return make_proxy_response(resp)

    if is_github_model(model_name):
        model_name = _canonical_gh_name(model_name)
        if not GITHUB_TOKEN:
            return jsonify(
                {
                    "error": {
                        "message": "GITHUB_TOKEN is not configured.",
                        "type": "api_error",
                    }
                }
            ), 500

        if do_stream:
            return Response(
                stream_with_context(
                    _github_chat_streaming_openai(model_name, messages)
                ),
                status=200,
                content_type="text/event-stream",
            )

        result = _github_chat_blocking(model_name, messages)
        if "error" in result:
            return jsonify(
                {"error": {"message": result["error"], "type": "api_error"}}
            ), 502

        return jsonify(
            {
                "id": "chatcmpl-github",
                "object": "chat.completion",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": result["message"],
                        "finish_reason": result.get("done_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0)
                    + result.get("eval_count", 0),
                },
            }
        )

    # Unknown model — return a clear 404 in OpenAI error format
    log.warning("/v1/chat/completions unknown model %r", model_name)
    return jsonify(
        {
            "error": {
                "message": f"Model '{model_name}' not found. Use GET /v1/models to list available models.",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
    ), 404


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting Ollama Proxy Server")
    log.info("Proxying Ollama to    : %s", REMOTE_OLLAMA_URL)
    log.info("GitHub Models         : %s", GITHUB_INFERENCE_ENDPOINT)
    log.info(
        "GITHUB_TOKEN          : %s",
        "configured" if GITHUB_TOKEN else "NOT SET — GH | models will return 500",
    )
    gc_count = sum(1 for m in GITHUB_MODELS if m.get("token") == "copilot")
    gh_count = sum(1 for m in GITHUB_MODELS if m.get("token") != "copilot")
    log.info("GH | models           : %d", gh_count)
    log.info(
        "GC | models           : %s",
        f"{gc_count} available" if gc_count else "0 — run GET /auth/copilot to log in",
    )
    log.info("Listening on          : http://0.0.0.0:%d", PORT)
    # debug=False is intentional — debug mode returns HTML error pages
    app.run(host="0.0.0.0", port=PORT, debug=False)
