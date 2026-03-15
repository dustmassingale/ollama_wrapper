#!/usr/bin/env python3
"""
zed_agent_service.py

FastAPI MVP for a standalone local RAG agent service intended to integrate with Zed.
- Basic in-memory document store (DOC_STORE)
- /embed proxies to an upstream model proxy
- /index stores documents locally
- /search performs a naive substring search (placeholder for vector DB)
- /chat assembles a compact RAG prompt and proxies chat/completion calls to upstream model proxy
- /apply returns a simulated apply result (Zed extension should apply patches locally after user Allow)

Usage:
  - Set environment variables as needed:
      LOCAL_PROXY_BASE (default: http://127.0.0.1:5000/v1)
      DISABLE_SSL_VERIFY ("true" to disable SSL verification to upstream)
      EMBED_MODEL (default: text-embedding-3-small)
      CHAT_MODEL (default: gpt-4o)
      PORT (default: 7860)
  - Run with uvicorn:
      uvicorn zed_agent_service:app --reload --port 7860
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
PROXY_BASE = os.getenv("LOCAL_PROXY_BASE", "http://127.0.0.1:5000/v1").rstrip("/")
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "false").strip().lower() == "true"
SSL_VERIFY = not DISABLE_SSL_VERIFY
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
DEFAULT_PORT = int(os.getenv("PORT", "7860"))

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("zed_agent_service")
logger.info("Zed Agent Service starting")
logger.info("Model proxy base: %s", PROXY_BASE)
logger.info("SSL verification enabled: %s", SSL_VERIFY)

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Zed Local Agent Service (MVP)", version="0.1.0")


# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class EmbedRequest(BaseModel):
    input: List[str] = Field(..., example=["text to embed"])


class IndexRequest(BaseModel):
    id: Optional[str] = Field(
        None, description="Optional document id. Generated if not provided."
    )
    text: str = Field(..., description="Document text to index")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of top results to return")


class ChatMessage(BaseModel):
    role: str = Field(..., description="system|user|assistant")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: int = Field(
        5, description="Number of docs to retrieve and include as context"
    )
    model: Optional[str] = Field(None, description="Override default chat model")


class ApplyRequest(BaseModel):
    file_path: str
    patch: str
    dry_run: bool = True


# ------------------------------------------------------------
# In-memory stores (MVP)
# ------------------------------------------------------------
# DOC_STORE: id -> { "text": str, "metadata": dict }
DOC_STORE: Dict[str, Dict[str, Any]] = {}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _proxy_post(path: str, json_payload: dict, timeout: int = 60) -> requests.Response:
    """
    Post to the configured model proxy. Raises HTTPException on network problems.
    """
    url = f"{PROXY_BASE.rstrip('/')}/{path.lstrip('/')}"
    try:
        resp = requests.post(url, json=json_payload, timeout=timeout, verify=SSL_VERIFY)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        logger.exception("Request to model proxy failed: %s %s", url, e)
        # Wrap as HTTPException for FastAPI to return JSON error
        raise HTTPException(status_code=502, detail=f"Model proxy request failed: {e}")


def _naive_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Naive substring-count scoring search across DOC_STORE.
    Returns list of hits: {id, score, text, metadata}
    """
    q = query.lower().strip()
    if not q:
        return []
    hits = []
    for doc_id, doc in DOC_STORE.items():
        txt = doc.get("text", "")
        score = txt.lower().count(q)
        # Also add a tiny length-normalized boost for short exact matches
        if score == 0:
            if q in txt.lower():
                score = 1
        if score:
            hits.append(
                {
                    "id": doc_id,
                    "score": score,
                    "text": txt,
                    "metadata": doc.get("metadata", {}),
                }
            )
    hits.sort(key=lambda d: d["score"], reverse=True)
    return hits[:top_k]


def _compact_snippet(text: str, max_chars: int = 2000) -> str:
    """
    Return a compact snippet for inclusion in prompts: clamp to max_chars with ellipsis.
    """
    if len(text) <= max_chars:
        return text
    # try to keep beginning and end
    head = text[: max_chars // 2].rstrip()
    tail = text[-(max_chars // 2) :].lstrip()
    return head + "\n\n...[snip]...\n\n" + tail


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    """Health check"""
    return {"ok": True, "note": "Zed Local Agent Service (MVP)"}


@app.post("/embed", tags=["models"])
def embed(req: EmbedRequest):
    """
    Proxy embeddings to the configured model proxy.
    Returns the full JSON response from the proxy.
    """
    payload = {"input": req.input, "model": EMBED_MODEL}
    logger.info("Embedding request: %d items, model=%s", len(req.input), EMBED_MODEL)
    resp = _proxy_post("/embeddings", payload, timeout=30)
    try:
        return resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Invalid JSON from proxy /embeddings"
        )


@app.post("/index", tags=["store"])
def index(req: IndexRequest):
    """
    Index a document into the in-memory store (MVP).
    Returns the assigned id.
    """
    doc_id = req.id or str(uuid.uuid4())
    DOC_STORE[doc_id] = {"text": req.text, "metadata": req.metadata or {}}
    logger.info("Indexed doc id=%s (len=%d chars)", doc_id, len(req.text))
    return {"ok": True, "id": doc_id}


@app.post("/search", tags=["store"])
def search(req: SearchRequest):
    """
    Naive search (substring-count) over in-memory documents.
    """
    results = _naive_search(req.query, top_k=req.top_k)
    logger.info("Search for '%s' returned %d hit(s)", req.query, len(results))
    return {"results": results}


@app.post("/chat", tags=["agent"])
def chat(req: ChatRequest):
    """
    RAG-style chat endpoint:
    - retrieve top_k docs (via naive search for MVP)
    - append compact retrieved docs as system context
    - call the upstream chat/completions endpoint on the configured proxy
    - return the proxy's JSON response
    """
    logger.info(
        "Chat request: messages=%d top_k=%d model_override=%s",
        len(req.messages),
        req.top_k,
        req.model,
    )
    # 1) retrieve hits
    last_user_content = ""
    if req.messages:
        # find last user message content
        for m in reversed(req.messages):
            if m.role == "user":
                last_user_content = m.content
                break
        if not last_user_content:
            # fallback: last message
            last_user_content = req.messages[-1].content

    hits = []
    if req.top_k > 0 and last_user_content:
        hits = _naive_search(last_user_content, top_k=req.top_k)

    # 2) build context parts
    context_parts: List[str] = []
    for h in hits:
        snippet = _compact_snippet(h["text"], max_chars=1500)
        context_parts.append(f"=== DOC {h['id']} (score={h['score']}) ===\n{snippet}\n")

    # 3) assemble messages for proxy
    system_msg = {
        "role": "system",
        "content": "You are an assistant that should use the provided retrieved documents when applicable. Answer concisely.",
    }
    assembled_messages = [system_msg]
    # include user's provided system messages and conversation
    for m in req.messages:
        assembled_messages.append({"role": m.role, "content": m.content})

    if context_parts:
        assembled_messages.append(
            {
                "role": "system",
                "content": "Retrieved documents:\n" + "\n".join(context_parts),
            }
        )

    # 4) call upstream proxy chat/completions
    model_to_use = req.model or CHAT_MODEL
    payload = {
        "model": model_to_use,
        "messages": assembled_messages,
        # optional: caller may tune generation params via env or extended API
        "max_tokens": 1024,
    }

    # Prefer chat/completions path; some proxies accept different paths; adjust as needed.
    try:
        resp = _proxy_post("/chat/completions", payload, timeout=120)
    except HTTPException:
        # fallback to /chat
        try:
            resp = _proxy_post("/chat", payload, timeout=120)
        except HTTPException as e:
            raise e

    try:
        result_json = resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Invalid JSON returned from proxy chat endpoint"
        )

    # Attach retrieved-hit summary in envelope for clients to show plan/context
    envelope = {"retrieved": hits, "proxy_response": result_json}
    logger.info("Chat completed, proxy returned keys: %s", list(result_json.keys()))
    return envelope


@app.post("/apply", tags=["agent"])
def apply(req: ApplyRequest):
    """
    Return a simulated apply result. The Zed extension should apply the patch locally
    after the user explicitly allows the change. This endpoint returns the patch and an
    'applied' boolean depending on dry_run flag.
    """
    logger.info("Apply called for file=%s dry_run=%s", req.file_path, req.dry_run)
    applied = False if req.dry_run else True
    return {"ok": True, "applied": applied, "file": req.file_path, "patch": req.patch}


@app.get("/docs_index", tags=["store"])
def docs_index():
    """Return a small listing of current indexed docs (for debugging)."""
    return {"count": len(DOC_STORE), "docs": list(DOC_STORE.keys())}


# ------------------------------------------------------------
# If run as script, print helpful startup information (uvicorn recommended)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Running via `python zed_agent_service.py` is supported for quick tests, but use uvicorn in production/dev.
    import uvicorn

    logger.info("Starting uvicorn on 0.0.0.0:%d (DEV only)", DEFAULT_PORT)
    uvicorn.run("zed_agent_service:app", host="0.0.0.0", port=DEFAULT_PORT, reload=True)
