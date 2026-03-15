"""
Ollama Proxy Server with real GitHub Models support.

- Ollama API requests are proxied to the remote server at REMOTE_OLLAMA_URL.
- GitHub model requests are fulfilled via the azure-ai-inference SDK against
  https://models.inference.ai.azure.com using a GITHUB_TOKEN env var.
- Responses are translated into Ollama-format NDJSON so any Ollama client
  (Continue.dev, Open WebUI, etc.) works transparently.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

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
GITHUB_TOKEN_COPILOT = os.getenv("GITHUB_TOKEN_COPILOT", "")
GITHUB_INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

# ---------------------------------------------------------------------------
# Model discovery
#
# On startup we query the live /models endpoint for each configured token and
# build the catalogue dynamically. The hardcoded list below is only used as a
# fallback when the discovery request fails (e.g. no network, bad token).
# ---------------------------------------------------------------------------

_FALLBACK_STANDARD = [
    {"sdk_name": "gpt-4o", "family": "gpt", "size": "unknown"},
    {"sdk_name": "gpt-4o-mini", "family": "gpt", "size": "unknown"},
    {"sdk_name": "gpt-4.1", "family": "gpt", "size": "unknown"},
    {"sdk_name": "gpt-4.1-mini", "family": "gpt", "size": "unknown"},
    {"sdk_name": "Meta-Llama-3.1-405B-Instruct", "family": "llama", "size": "405B"},
    {"sdk_name": "Meta-Llama-3.1-8B-Instruct", "family": "llama", "size": "8B"},
]

_FALLBACK_COPILOT: list[dict] = []  # nothing confirmed yet; populated by discovery


def _discover_models(token: str, prefix: str, token_label: str) -> list[dict]:
    """
    Query the GitHub Models catalogue endpoint and return a list of proxy
    model entries. Falls back to the hardcoded list on any error.
    """
    if not token:
        return []

    fallback = _FALLBACK_STANDARD if token_label == "standard" else _FALLBACK_COPILOT

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

# Populated at module load time by querying the live /models endpoint.
GITHUB_MODELS: list[dict] = _discover_models(
    GITHUB_TOKEN, GH_PREFIX, "standard"
) + _discover_models(GITHUB_TOKEN_COPILOT, GC_PREFIX, "copilot")

# Fast lookup: display name -> catalogue entry
GITHUB_MODEL_MAP: dict[str, dict] = {m["name"]: m for m in GITHUB_MODELS}


def _active_github_models() -> list[dict]:
    """Return all discovered models (discovery already filtered by token availability)."""
    return GITHUB_MODELS


def _token_for_model(name: str) -> str:
    """Return the appropriate token string for the given (canonical) model name."""
    entry = GITHUB_MODEL_MAP.get(name, {})
    if entry.get("token") == "copilot":
        if not GITHUB_TOKEN_COPILOT:
            raise RuntimeError(
                f"'{name}' requires GITHUB_TOKEN_COPILOT, which is not set."
            )
        return GITHUB_TOKEN_COPILOT
    if not GITHUB_TOKEN:
        raise RuntimeError(f"'{name}' requires GITHUB_TOKEN, which is not set.")
    return GITHUB_TOKEN


def is_github_model(name: str) -> bool:
    """Accept prefixed, bare, and dot-notation names (e.g. 'claude-haiku-4.5')."""
    return _canonical_gh_name(name) in GITHUB_MODEL_MAP


def _canonical_gh_name(name: str) -> str:
    """Return the prefixed display name regardless of whether the prefix was supplied.

    Handles these input forms:
      - "GH | gpt-4o"            (already canonical, standard)
      - "GC | claude-haiku-4-5"  (already canonical, copilot)
      - "claude-haiku-4-5"       (bare, dashes)
      - "claude-haiku-4.5"       (bare, dots — as sent by Continue.dev)
    """
    if name in GITHUB_MODEL_MAP:
        return name
    # Try both prefixes as-is
    for prefix in (GH_PREFIX, GC_PREFIX):
        prefixed = prefix + name
        if prefixed in GITHUB_MODEL_MAP:
            return prefixed
    # Normalise dots to dashes and try again (e.g. "claude-haiku-4.5" -> "claude-haiku-4-5")
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
    """Build a ChatCompletionsClient using the correct token for the model."""
    token = _token_for_model(model_name)
    return ChatCompletionsClient(
        endpoint=GITHUB_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential(token),
    )


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
        entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(model_name), {})
        required_token = (
            GITHUB_TOKEN_COPILOT if entry.get("token") == "copilot" else GITHUB_TOKEN
        )
        if not required_token:
            token_var = (
                "GITHUB_TOKEN_COPILOT"
                if entry.get("token") == "copilot"
                else "GITHUB_TOKEN"
            )
            err = {
                "error": f"'{model_name}' requires {token_var}, which is not set in your .env."
            }
            return jsonify(err), 500

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
        entry = GITHUB_MODEL_MAP.get(model_name, {})
        required_token = (
            GITHUB_TOKEN_COPILOT if entry.get("token") == "copilot" else GITHUB_TOKEN
        )
        if not required_token:
            token_var = (
                "GITHUB_TOKEN_COPILOT"
                if entry.get("token") == "copilot"
                else "GITHUB_TOKEN"
            )
            err = {
                "error": f"'{model_name}' requires {token_var}, which is not set in your .env."
            }
            return jsonify(err), 500

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
        entry = GITHUB_MODEL_MAP.get(model_name, {})
        required_token = (
            GITHUB_TOKEN_COPILOT if entry.get("token") == "copilot" else GITHUB_TOKEN
        )
        if not required_token:
            token_var = (
                "GITHUB_TOKEN_COPILOT"
                if entry.get("token") == "copilot"
                else "GITHUB_TOKEN"
            )
            return jsonify(
                {
                    "error": f"'{model_name}' requires {token_var}, which is not set in your .env."
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


@app.route("/health", methods=["GET"])
def health():
    try:
        resp = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        remote_status = "ok" if resp.status_code == 200 else "error"
    except Exception:
        remote_status = "error"

    github_status = {
        "standard": "ready" if GITHUB_TOKEN else "not set — add GITHUB_TOKEN to .env",
        "copilot": "ready"
        if GITHUB_TOKEN_COPILOT
        else "not set — add GITHUB_TOKEN_COPILOT to .env (optional)",
    }
    status = "healthy" if remote_status == "ok" else "degraded"

    return jsonify(
        {
            "status": status,
            "remote_ollama": remote_status,
            "github_models": github_status,
        }
    )


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "message": "Ollama Proxy with GitHub Models",
            "version": "1.4.0",
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
        entry = GITHUB_MODEL_MAP.get(model_name, {})
        required_token = (
            GITHUB_TOKEN_COPILOT if entry.get("token") == "copilot" else GITHUB_TOKEN
        )
        if not required_token:
            token_var = (
                "GITHUB_TOKEN_COPILOT"
                if entry.get("token") == "copilot"
                else "GITHUB_TOKEN"
            )
            return jsonify(
                {
                    "error": {
                        "message": f"'{model_name}' requires {token_var}, which is not set in your .env.",
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
        "configured"
        if GITHUB_TOKEN
        else "NOT SET — standard GH models will return 500",
    )
    log.info(
        "GITHUB_TOKEN_COPILOT  : %s",
        "configured"
        if GITHUB_TOKEN_COPILOT
        else "not set — Copilot-tier models hidden",
    )
    log.info(
        "Active GitHub models  : %d / %d",
        len(_active_github_models()),
        len(GITHUB_MODELS),
    )
    log.info("Listening on       : http://0.0.0.0:%d", PORT)
    # debug=False is intentional — debug mode returns HTML error pages
    app.run(host="0.0.0.0", port=PORT, debug=False)
