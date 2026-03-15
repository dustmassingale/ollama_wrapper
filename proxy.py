"""
Simple Ollama Proxy Server with GitHub Models for Copilot
Routes requests to 192.168.1.155 and adds GitHub models
"""

import json
import logging
import sys
from datetime import datetime

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REMOTE_OLLAMA_URL = "http://192.168.1.155:11434"
PORT = 11434

# ---------------------------------------------------------------------------
# GitHub / Copilot model catalogue
# ---------------------------------------------------------------------------

GITHUB_MODELS = [
    {
        "name": "gpt-5-mini",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "claude-haiku-4.5",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "claude-opus-4.6",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "claude-sonnet-4.6",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "gemini-3-flash",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "gemini-3.1-pro",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "gpt-5.3-codex",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "gpt-5.4",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
    {
        "name": "grok-code-fast-1",
        "size": 0,
        "modified_at": "2024-01-01T00:00:00Z",
        "details": {},
    },
]

GITHUB_MODEL_NAMES = {m["name"] for m in GITHUB_MODELS}


def is_github_model(name: str) -> bool:
    return name in GITHUB_MODEL_NAMES


# ---------------------------------------------------------------------------
# Global error handlers — always return JSON, never HTML
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
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """Yield raw bytes from an upstream streaming response."""
    for chunk in upstream_response.iter_content(chunk_size=None):
        if chunk:
            yield chunk


def make_proxy_response(
    resp, default_content_type: str = "application/json"
) -> Response:
    """Wrap an upstream requests.Response into a Flask Response."""
    return Response(
        resp.content,
        status=resp.status_code,
        content_type=resp.headers.get("Content-Type", default_content_type),
    )


def make_streaming_proxy_response(
    resp, default_content_type: str = "application/x-ndjson"
) -> Response:
    """Wrap an upstream streaming requests.Response into a Flask streaming Response."""
    return Response(
        stream_with_context(stream_chunks(resp)),
        status=resp.status_code,
        content_type=resp.headers.get("Content-Type", default_content_type),
    )


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

    return jsonify({"models": remote_models + GITHUB_MODELS})


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
        return jsonify(
            {
                "license": "",
                "modelfile": f"# GitHub / Copilot model: {model_name}",
                "parameters": "",
                "template": "{{ .Prompt }}",
                "details": {
                    "format": "gguf",
                    "family": "github",
                    "families": ["github"],
                    "parameter_size": "unknown",
                    "quantization_level": "unknown",
                },
                "model_info": {},
            }
        )

    resp = proxy_request("POST", "/api/show", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/chat  (Ollama-native NDJSON chat)
# ---------------------------------------------------------------------------


def _github_chat_payload(model_name: str) -> dict:
    return {
        "model": model_name,
        "created_at": now_iso(),
        "message": {
            "role": "assistant",
            "content": (
                f"'{model_name}' is listed in this proxy's model catalogue "
                "but is not yet connected to a live backend. "
                "Please configure a real endpoint for this model."
            ),
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "eval_count": 0,
    }


@app.route("/api/chat", methods=["POST"])
def chat():
    """Ollama /api/chat — proxy to remote or return synthetic GitHub response."""
    data = request.json or {}
    model_name = data.get("model", "")
    # Continue.dev and most clients send "stream": true by default for /api/chat
    do_stream = data.get("stream", True)
    log.debug("/api/chat model=%r stream=%s", model_name, do_stream)

    if is_github_model(model_name):
        payload = _github_chat_payload(model_name)
        if do_stream:
            # Streaming NDJSON: single line then done
            body = json.dumps(payload) + "\n"
        else:
            # Non-streaming: plain JSON object
            body = json.dumps(payload)
        return Response(body, status=200, content_type="application/json")

    resp = proxy_request("POST", "/api/chat", data=data, stream=do_stream)
    if do_stream:
        return make_streaming_proxy_response(resp, "application/x-ndjson")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/generate  (Ollama-native NDJSON generate)
# ---------------------------------------------------------------------------


def _github_generate_payload(model_name: str) -> dict:
    return {
        "model": model_name,
        "created_at": now_iso(),
        "response": (
            f"'{model_name}' is listed in this proxy's model catalogue "
            "but is not yet connected to a live backend. "
            "Please configure a real endpoint for this model."
        ),
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "eval_count": 0,
    }


@app.route("/api/generate", methods=["POST"])
def generate():
    """Ollama /api/generate — proxy to remote or return synthetic GitHub response."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", True)
    log.debug("/api/generate model=%r stream=%s", model_name, do_stream)

    if is_github_model(model_name):
        payload = _github_generate_payload(model_name)
        if do_stream:
            body = json.dumps(payload) + "\n"
        else:
            body = json.dumps(payload)
        return Response(body, status=200, content_type="application/json")

    resp = proxy_request("POST", "/api/generate", data=data, stream=do_stream)
    if do_stream:
        return make_streaming_proxy_response(resp, "application/x-ndjson")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/chat/completions  (OpenAI-compat)
# ---------------------------------------------------------------------------


@app.route("/api/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions — proxy to remote Ollama."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", False)
    log.debug("/api/chat/completions model=%r stream=%s", model_name, do_stream)

    if is_github_model(model_name):
        return jsonify(
            {
                "id": "chatcmpl-github",
                "object": "chat.completion",
                "created": int(datetime.utcnow().timestamp()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                f"'{model_name}' is listed in this proxy's model catalogue "
                                "but is not yet connected to a live backend."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    resp = proxy_request("POST", "/api/chat/completions", data=data, stream=do_stream)
    if do_stream:
        return make_streaming_proxy_response(resp, "text/event-stream")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/embeddings
# ---------------------------------------------------------------------------


@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    """Proxy embeddings requests to remote Ollama."""
    data = request.json or {}
    log.debug("/api/embeddings model=%r", data.get("model"))
    resp = proxy_request("POST", "/api/embeddings", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/embed  (newer Ollama alias)
# ---------------------------------------------------------------------------


@app.route("/api/embed", methods=["POST"])
def embed():
    """Proxy /api/embed requests to remote Ollama."""
    data = request.json or {}
    log.debug("/api/embed model=%r", data.get("model"))
    resp = proxy_request("POST", "/api/embed", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/pull
# ---------------------------------------------------------------------------


@app.route("/api/pull", methods=["POST"])
def pull():
    """Proxy pull requests to remote Ollama (streamed progress)."""
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
    """List running models on the remote Ollama server."""
    resp = proxy_request("GET", "/api/ps")
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/delete
# ---------------------------------------------------------------------------


@app.route("/api/delete", methods=["DELETE"])
def delete():
    """Proxy model deletion to the remote Ollama server."""
    data = request.json or {}
    resp = proxy_request("DELETE", "/api/delete", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /api/copy
# ---------------------------------------------------------------------------


@app.route("/api/copy", methods=["POST"])
def copy():
    """Proxy model copy to the remote Ollama server."""
    data = request.json or {}
    resp = proxy_request("POST", "/api/copy", data=data)
    return make_proxy_response(resp)


# ---------------------------------------------------------------------------
# /health  and  /
# ---------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        resp = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        remote_status = "ok" if resp.status_code == 200 else "error"
    except Exception:
        remote_status = "error"

    status = "healthy" if remote_status == "ok" else "degraded"
    return jsonify(
        {"status": status, "remote_ollama": remote_status, "github_models": "available"}
    )


@app.route("/", methods=["GET"])
def root():
    """Root endpoint."""
    return jsonify(
        {
            "message": "Ollama Proxy with GitHub Models",
            "version": "1.2.0",
            "endpoints": [
                "GET    /api/tags             — list all models",
                "POST   /api/show             — model metadata",
                "POST   /api/chat             — Ollama-style chat (NDJSON)",
                "POST   /api/generate         — Ollama-style generate (NDJSON)",
                "POST   /api/chat/completions — OpenAI-compat chat completions",
                "POST   /api/embeddings       — embeddings (legacy)",
                "POST   /api/embed            — embeddings",
                "POST   /api/pull             — pull a model",
                "GET    /api/ps              — running models",
                "DELETE /api/delete          — delete a model",
                "POST   /api/copy            — copy a model",
                "GET    /health              — health check",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Starting Ollama Proxy Server")
    log.info("Proxying to : %s", REMOTE_OLLAMA_URL)
    log.info("GitHub models: %d available", len(GITHUB_MODELS))
    log.info("Listening on  http://0.0.0.0:%d", PORT)
    # debug=False is intentional — debug mode swallows errors into HTML pages
    app.run(host="0.0.0.0", port=PORT, debug=False)
