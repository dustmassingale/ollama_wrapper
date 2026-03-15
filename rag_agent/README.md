# Zed Local RAG Agent — MVP Scaffold

Version: 0.1.0  
Date: 2026-03-15

This repository folder contains a standalone Retrieval-Augmented Generation (RAG) agent service scaffold designed to integrate with the Zed editor. It is intentionally lightweight and local-first — the service performs retrieval, prompt assembly, and proxies embedding/chat calls to your existing model proxy (default: `http://127.0.0.1:5000/v1`). A minimal Zed extension skeleton demonstrates how Zed can invoke the agent and present an allow/deny workflow for applying patches.

This folder contains:
- `zed_agent_service.py` — FastAPI MVP service (indexing, naive search, embed proxying, chat proxying, apply stub)
- `requirements.txt` — pinned dependencies for the MVP
- `memory.json` — seed memory for the agent (user preferences & project hints)
- `.env.example` — example environment variables
- `SRD.md` — software requirements document (design, API, roadmap)
- `zed-extension/` — skeleton Zed extension manifest and TypeScript handler
  - `extension.yaml`
  - `src/extension.ts`

Goals
- Reduce tokens sent to heavy models by retrieving relevant context locally
- Keep the editor workflow (Zed) snappy with an allow/deny UX for edits
- Be easy to run locally and extend later (add Chroma/FAISS, reranker, streaming)

---

## Quick start (local dev)

1. Create a new directory and copy these files (or clone a repo that contains this scaffold).
2. Create and activate a virtual environment:
   - macOS / Linux
     - python -m venv .venv
     - source .venv/bin/activate
   - Windows (PowerShell)
     - python -m venv .venv
     - .\.venv\Scripts\Activate.ps1
3. Install dependencies:
   - pip install -r requirements.txt
4. Configure environment (optional)
   - Copy `.env.example` to `.env` or set environment variables directly:
     - `LOCAL_PROXY_BASE` — URL of your model proxy (default: `http://127.0.0.1:5000/v1`)
     - `DISABLE_SSL_VERIFY` — set to `true` for self-signed cert testing (not recommended for production)
     - `EMBED_MODEL` — embedding model (default: `text-embedding-3-small`)
     - `CHAT_MODEL` — default chat model (default: `gpt-4o`)
     - `PORT` — port for this service (default: `7860`)
5. Run the service:
   - uvicorn zed_agent_service:app --reload --port 7860
6. Test endpoints (you can use curl / HTTP client):
   - Index a doc:
     - curl -X POST "http://127.0.0.1:7860/index" -H "Content-Type: application/json" -d "{\"text\":\"Example document text to index\"}"
   - Search:
     - curl -X POST "http://127.0.0.1:7860/search" -H "Content-Type: application/json" -d "{\"query\":\"Example\", \"top_k\":5}"
   - Chat:
     - curl -X POST "http://127.0.0.1:7860/chat" -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Summarize the example\"}], \"top_k\":3}"

---

## How it works (MVP)

- The agent service is a small HTTP API that:
  - Accepts documents via `/index` and stores them in an in-memory store (MVP).
  - Offers a `/search` endpoint using a naive substring-count scorer (placeholder for vector store).
  - Proxies embedding requests to your model proxy via `/embed`.
  - Builds a compact RAG prompt (retrieved doc snippets appended as a system message) and proxies chat requests to the model proxy via `/chat`.
  - Returns structured responses including retrieved-hit metadata so the Zed extension can show the plan and the source context.
  - Provides an `/apply` endpoint that returns a simulated application result — the Zed extension should apply the patch locally after user Allow.

- Token-savings strategy:
  - Send only selected region(s) and a compact snippet of retrieved docs (not whole repo).
  - Use embeddings + vector DB in future iterations so the model receives only the top-K relevant contexts.

---

## Zed extension (skeleton)

The `zed-extension` folder contains a starting manifest and a TypeScript skeleton. The extension demonstrates:
- Registering a command in Zed (for example: `zed-rag-agent.ask`).
- Collecting context from Zed (selected text / file path / small surrounding lines).
- Posting a `/chat` request to the local agent service.
- Presenting the assistant response and a preview of suggested edits/patches.
- An allow/deny UI (MVP uses a simple modal/info prompt; expand this later).
- On Allow, the extension should apply changes (using Zed editing APIs) or call `/apply` and apply returned diff.

Local install path for Zed extensions (platforms):
- macOS: `~/Library/Application Support/Zed/extensions/`
- Linux: `$XDG_DATA_HOME/zed/extensions` or `~/.local/share/zed/extensions`
- Windows: `%LOCALAPPDATA%\Zed\extensions`

To test the extension:
- Place the manifest and built JS in the local extensions directory for Zed and restart Zed.
- The skeleton uses fetch to call `http://127.0.0.1:7860` — adjust env if needed.

Note: Zed extension APIs are evolving — the provided code is intentionally minimal to act as a starting point.

---

## Configuration & env variables

Recommended environment variables (see `.env.example`):
- LOCAL_PROXY_BASE — upstream model proxy base (e.g., `http://127.0.0.1:5000/v1`)
- DISABLE_SSL_VERIFY — set to `true` to disable TLS verification (only temporary/troubleshooting)
- EMBED_MODEL — embedding model name used when calling the proxy
- CHAT_MODEL — default chat model name to request from the proxy
- PORT — agent service port (default 7860)
- STORE_BACKEND — `memory` | `chroma` | `faiss` | `hnswlib` (MVP uses `memory`)

When you later add vector DB support, populate configuration to point to the DB or use a local Chroma instance.

---

## Next steps (recommended roadmap)

1. Replace in-memory store with a vector DB:
   - Chroma + hnswlib (easy Python option)
   - Faiss (higher performance on larger corpora)
2. Add an incremental indexer:
   - File-system watcher to index changed files automatically
   - Respect `.gitignore`/project-specific ignores
3. Add a reranker:
   - Cross-encoder model or use the chat model to rerank top-K
4. Stream assistant output to Zed:
   - Use SSE or websockets for progressive results and faster UX
5. Harden security:
   - Add local API token guard if you expose this beyond localhost
   - Disable `DISABLE_SSL_VERIFY` in all production workflows
6. Create richer Zed UI:
   - Diff viewer, suggestions panel, history, rollback support

---

## Development & testing tips

- Use `curl` or Postman to test API surfaces before integrating with Zed.
- Index small, targeted documents first to validate retrieval and prompt assembly.
- Experiment with `EMBED_MODEL` and `CHAT_MODEL` values using your model proxy to find the best trade-offs for speed and cost.
- Keep `DISABLE_SSL_VERIFY` off unless absolutely necessary.

---

## Files of interest

- `zed_agent_service.py` — main FastAPI app (start here)
- `requirements.txt` — dependencies
- `memory.json` — seed memory for models and the extension
- `SRD.md` — software requirements & API specification
- `zed-extension/extension.yaml` — extension manifest
- `zed-extension/src/extension.ts` — extension skeleton code

---

## Contributing

This scaffold is intended as a starting point. If you plan to extend it:
- Follow the roadmap above and add small, testable features per PR.
- Keep secrets out of source control — use `.env` and environment variables.
- Write unit tests for indexing/search and integration tests for proxy handling.

---

## License & acknowledgements

Add your preferred license file at the repo root (e.g., `LICENSE`) and attribution for third-party libraries used (FastAPI, Uvicorn, Requests, etc.).

---

If you'd like, I can produce:
- A Chroma-backed version of the service (indexing, persistence, vector search)
- A more complete Zed extension with UI for allow/deny and patch application
- CI configuration and example unit/integration tests

Tell me which of the above you'd like next and I will produce the files to paste into a new repository.