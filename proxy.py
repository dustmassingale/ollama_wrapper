"""
Simple Ollama Proxy Server with GitHub Models for Copilot
Routes requests to 192.168.1.155 and adds GitHub models
"""

import json

import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# Configuration
REMOTE_OLLAMA_URL = "http://192.168.1.155:11434"

# GitHub models for Copilot integration
GITHUB_MODELS = [
    {"name": "gpt-5-mini", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "claude-haiku-4.5", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "claude-opus-4.6", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "claude-sonnet-4.6", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "gemini-3-flash", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "gemini-3.1-pro", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "gpt-5.3-codex", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "gpt-5.4", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "grok-code-fast-1", "size": 0, "modified_at": "2024-01-01T00:00:00Z"},
]


@app.route("/api/tags", methods=["GET"])
def get_tags():
    """Get all models - both from Ollama and GitHub"""
    try:
        # Get models from remote Ollama server
        response = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        remote_models = response.json().get("models", [])
    except Exception as e:
        print(f"Error fetching remote models: {e}")
        remote_models = []

    # Combine with GitHub models
    all_models = remote_models + GITHUB_MODELS

    return jsonify({"models": all_models})


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate text - proxy to remote Ollama or handle GitHub models"""
    data = request.json
    model_name = data.get("model")

    # Check if it's a GitHub model
    if any(m["name"] == model_name for m in GITHUB_MODELS):
        return jsonify(
            {
                "model": model_name,
                "created_at": "2024-01-01T00:00:00Z",
                "response": f"GitHub model {model_name} is available via API",
                "done": True,
            }
        )

    # Otherwise proxy to remote Ollama
    try:
        response = requests.post(
            f"{REMOTE_OLLAMA_URL}/api/generate",
            json=data,
            timeout=300,
            stream=data.get("stream", False),
        )

        if data.get("stream"):
            return Response(
                response.iter_content(chunk_size=1024), content_type="application/json"
            )
        else:
            return response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/completions", methods=["POST"])
def chat():
    """Chat completions - proxy to remote Ollama or handle GitHub models"""
    data = request.json
    model_name = data.get("model")

    # Check if it's a GitHub model
    if any(m["name"] == model_name for m in GITHUB_MODELS):
        return jsonify(
            {
                "model": model_name,
                "created_at": "2024-01-01T00:00:00Z",
                "response": f"GitHub model {model_name} is available via API",
                "done": True,
            }
        )

    # Otherwise proxy to remote Ollama
    try:
        response = requests.post(
            f"{REMOTE_OLLAMA_URL}/api/chat/completions",
            json=data,
            timeout=300,
            stream=data.get("stream", False),
        )

        if data.get("stream"):
            return Response(
                response.iter_content(chunk_size=1024), content_type="application/json"
            )
        else:
            return response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        response = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags", timeout=5)
        remote_status = "ok" if response.status_code == 200 else "error"
    except:
        remote_status = "error"

    status = "healthy" if remote_status == "ok" else "degraded"

    return jsonify(
        {"status": status, "remote_ollama": remote_status, "github_models": "available"}
    )


@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify(
        {
            "message": "Ollama Proxy with GitHub Models",
            "version": "1.0.0",
            "endpoints": [
                "/api/tags - List all models",
                "/api/generate - Generate text",
                "/api/chat/completions - Chat completion",
                "/health - Health check",
            ],
        }
    )


if __name__ == "__main__":
    print("Starting Ollama Proxy Server...")
    print(f"Proxying to: {REMOTE_OLLAMA_URL}")
    print(f"GitHub models: {len(GITHUB_MODELS)} models available")
    print("Running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
