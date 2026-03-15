"""
Simple Ollama Proxy Server with GitHub Models for Copilot
Routes requests to 192.168.1.155 and adds GitHub models
"""

import json

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

app = Flask(__name__)

# Configuration
REMOTE_OLLAMA_URL = "http://192.168.1.155:11434"

# GitHub models for Copilot integration
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


def is_github_model(name):
    return name in GITHUB_MODEL_NAMES


def proxy_stream(upstream_response):
    """Yield chunks from an upstream streaming response."""
    for chunk in upstream_response.iter_content(chunk_size=None):
        if chunk:
            yield chunk


def proxy_request(method, path, data=None, params=None, stream=False):
    """Forward a request to the remote Ollama server and return the response."""
    url = f"{REMOTE_OLLAMA_URL}{path}"
    headers = {"Content-Type": "application/json"}
    resp = requests.request(
        method,
        url,
        json=data,
        params=params,
        headers=headers,
        timeout=300,
        stream=stream,
    )
    return resp


# ---------------------------------------------------------------------------
# /api/tags — list all models
# ---------------------------------------------------------------------------


@app.route("/api/tags", methods=["GET"])
def get_tags():
    """Get all models — both from the remote Ollama server and GitHub."""
    try:
        resp = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        remote_models = resp.json().get("models", [])
    except Exception as e:
        print(f"Error fetching remote models: {e}")
        remote_models = []

    all_models = remote_models + GITHUB_MODELS
    return jsonify({"models": all_models})


# ---------------------------------------------------------------------------
# /api/show — model info
# ---------------------------------------------------------------------------


@app.route("/api/show", methods=["POST"])
def show():
    """Return model metadata. GitHub models get a synthetic response."""
    data = request.json or {}
    model_name = data.get("model") or data.get("name", "")

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

    try:
        resp = proxy_request("POST", "/api/show", data=data)
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/chat — Ollama-style chat (NDJSON, optionally streaming)
# ---------------------------------------------------------------------------


@app.route("/api/chat", methods=["POST"])
def chat():
    """Ollama /api/chat endpoint — proxy or synthetic GitHub response."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", True)

    if is_github_model(model_name):
        payload = json.dumps(
            {
                "model": model_name,
                "created_at": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": f"(GitHub model '{model_name}' is listed but not yet connected to a live backend.)",
                },
                "done": True,
                "done_reason": "stop",
            }
        )
        return Response(payload + "\n", status=200, content_type="application/x-ndjson")

    try:
        resp = proxy_request("POST", "/api/chat", data=data, stream=do_stream)
        if do_stream:
            return Response(
                stream_with_context(proxy_stream(resp)),
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "application/x-ndjson"),
            )
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/generate — Ollama-style completion (NDJSON, optionally streaming)
# ---------------------------------------------------------------------------


@app.route("/api/generate", methods=["POST"])
def generate():
    """Ollama /api/generate endpoint — proxy or synthetic GitHub response."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", True)

    if is_github_model(model_name):
        payload = json.dumps(
            {
                "model": model_name,
                "created_at": "2024-01-01T00:00:00Z",
                "response": f"(GitHub model '{model_name}' is listed but not yet connected to a live backend.)",
                "done": True,
            }
        )
        return Response(payload + "\n", status=200, content_type="application/x-ndjson")

    try:
        resp = proxy_request("POST", "/api/generate", data=data, stream=do_stream)
        if do_stream:
            return Response(
                stream_with_context(proxy_stream(resp)),
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "application/x-ndjson"),
            )
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/chat/completions — OpenAI-compat chat completions (pass-through)
# ---------------------------------------------------------------------------


@app.route("/api/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions — proxy to remote Ollama."""
    data = request.json or {}
    model_name = data.get("model", "")
    do_stream = data.get("stream", False)

    if is_github_model(model_name):
        return jsonify(
            {
                "id": "chatcmpl-github",
                "object": "chat.completion",
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"(GitHub model '{model_name}' is listed but not yet connected to a live backend.)",
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

    try:
        resp = proxy_request(
            "POST", "/api/chat/completions", data=data, stream=do_stream
        )
        if do_stream:
            return Response(
                stream_with_context(proxy_stream(resp)),
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "text/event-stream"),
            )
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/embeddings — proxy embeddings
# ---------------------------------------------------------------------------


@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    """Proxy embeddings requests to remote Ollama."""
    data = request.json or {}
    try:
        resp = proxy_request("POST", "/api/embeddings", data=data)
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/pull — proxy pull requests
# ---------------------------------------------------------------------------


@app.route("/api/pull", methods=["POST"])
def pull():
    """Proxy pull requests to remote Ollama (streamed progress)."""
    data = request.json or {}
    do_stream = data.get("stream", True)
    try:
        resp = proxy_request("POST", "/api/pull", data=data, stream=do_stream)
        if do_stream:
            return Response(
                stream_with_context(proxy_stream(resp)),
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "application/x-ndjson"),
            )
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/ps — running models
# ---------------------------------------------------------------------------


@app.route("/api/ps", methods=["GET"])
def ps():
    """List running models on the remote Ollama server."""
    try:
        resp = proxy_request("GET", "/api/ps")
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/delete — proxy delete
# ---------------------------------------------------------------------------


@app.route("/api/delete", methods=["DELETE"])
def delete():
    """Proxy model deletion to the remote Ollama server."""
    data = request.json or {}
    try:
        resp = proxy_request("DELETE", "/api/delete", data=data)
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/copy — proxy copy
# ---------------------------------------------------------------------------


@app.route("/api/copy", methods=["POST"])
def copy():
    """Proxy model copy to the remote Ollama server."""
    data = request.json or {}
    try:
        resp = proxy_request("POST", "/api/copy", data=data)
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# /health and /
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
        {
            "status": status,
            "remote_ollama": remote_status,
            "github_models": "available",
        }
    )


@app.route("/", methods=["GET"])
def root():
    """Root endpoint."""
    return jsonify(
        {
            "message": "Ollama Proxy with GitHub Models",
            "version": "1.1.0",
            "endpoints": [
                "GET  /api/tags              — list all models",
                "POST /api/show              — model metadata",
                "POST /api/chat              — Ollama-style chat (NDJSON)",
                "POST /api/generate          — Ollama-style generate (NDJSON)",
                "POST /api/chat/completions  — OpenAI-compat chat completions",
                "POST /api/embeddings        — embeddings",
                "POST /api/pull              — pull a model",
                "GET  /api/ps               — running models",
                "DELETE /api/delete         — delete a model",
                "POST /api/copy             — copy a model",
                "GET  /health               — health check",
            ],
        }
    )


if __name__ == "__main__":
    print("Starting Ollama Proxy Server...")
    print(f"Proxying to: {REMOTE_OLLAMA_URL}")
    print(f"GitHub models: {len(GITHUB_MODELS)} models available")
    print("Running on http://0.0.0.0:11434")
    app.run(host="0.0.0.0", port=11434, debug=True)
