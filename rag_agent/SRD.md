# SRD: Standalone Local RAG Agent for Zed

Version: 0.1.0  
Date: 2026-03-15  
Author: Engineering Team (you / repo owner)

---

## 1. Purpose

This document describes the Software Requirements Document (SRD) for a standalone Retrieval-Augmented Generation (RAG) Agent designed to work with the Zed editor. The service runs locally, handles embedding, retrieval, ranking, and orchestrates model calls through an existing model proxy (for Copilot, Bedrock, Ollama, GitHub models, etc.). A minimal Zed extension provides an allow/deny flow and applies edits returned by the agent.

Goals:
- Provide a local agent that dramatically reduces token usage by performing retrieval and ranking locally.
- Support configurable embedding model, vector store, and chat models via an upstream proxy (default: http://127.0.0.1:5000/v1/).
- Provide a small Zed extension to trigger agent workflows and apply patches conditionally (user allows).
- Be standalone — not integrated into the `ollama_wrapper` project. Files in `./rag_agent` are a self-contained scaffold to start a new repo.

Non-goals (v0 MVP):
- Production-grade vector database deployment. MVP uses an in-memory index and plain-text search; later we'll add Chroma/FAISS/hnswlib.
- Full security hardening for public access. MVP binds to localhost.
- Automatic repository file modifications without explicit user allow in Zed.

---

## 2. Stakeholders

- Primary user: Developer using Zed who wants a low-token, high-UX RAG assistant with allow/deny workflows.
- Maintainer: Owner of the standalone `rag_agent` repo.
- Integrations: Local model proxy operators (the model proxy already used in your environment).
- Future users: Teams who want on-device RAG tooling with Zed integration.

---

## 3. High-level Architecture

Components:
- Agent Service (FastAPI) — orchestrates retrieval, embedding, reranking, model calls through the model proxy, and exposes REST endpoints:
  - /embed — proxies embedding calls
  - /index — add doc to in-memory store (or persistent store)
  - /search — retrieve candidates
  - /chat — assemble RAG prompt, call chat model via proxy
  - /apply — (MVP) return patch or simulated apply result (Zed applies on allow)
- Model Proxy — existing local proxy at http://127.0.0.1:5000/v1/ that actually calls Copilot/Bedrock/OpenAI/Ollama
- Vector Store (optional in MVP) — in-memory dict or future Chroma/FAISS/hnswlib
- Zed Extension (TypeScript skeleton) — registers a command to call agent endpoints, shows allow/deny UI, and applies patches on Allow

Data flow:
1. User triggers an agent action in Zed (e.g., "refactor function" or "explain code region").
2. Zed extension sends context (selected code + metadata) to Agent Service `/chat`.
3. Agent Service retrieves top-K docs (local search or vector DB) and builds RAG prompt.
4. Agent Service calls chat endpoint on the Model Proxy (via /chat/completions) and returns structured response including plan and patches.
5. Zed displays the plan and asks the user to Allow/Deny. On Allow, extension posts to `/apply` or applies returned patch locally.

---

## 4. Functional Requirements

R1. Agent Service must accept embed, index, search, chat, and apply requests over HTTP (localhost).  
R2. Agent Service must proxy embedding and chat requests to configurable upstream model proxy.  
R3. Agent must return responses in an OpenAI-compatible shape for the Zed extension to show assistant text and structured plan (optional).  
R4. Zed extension must provide a command to call the agent and display results with an allow/deny modal.  
R5. On Allow, Zed must apply patches atomically to files in the workspace.  
R6. The agent must support configurable model names via environment variables.  
R7. All local storage (index/docs) must be optional and easily swappable for a real vector DB.

---

## 5. Non-functional Requirements

N1. Local-first: service binds to localhost only by default.  
N2. Low-latency for local retrieval and prompt assembly; heavy model work is proxied.  
N3. Safe-by-default: changes are not applied automatically; Zed must prompt user consent.  
N4. Easily extensible to add Chroma/FAISS/hnswlib as vector store backends.  
N5. Clear, developer-friendly logs to assist debugging.

---

## 6. API Specification (MVP)

All endpoints are JSON over HTTP(S). Default host: http://127.0.0.1:7860 (configurable).

- GET /
  - Description: health check
  - Response: { "ok": true, "note": "Zed Local Agent Service (MVP)" }

- POST /embed
  - Body:
    - input: [string]
  - Behavior: forwards to proxy `/embeddings` with model configurable by env EMBED_MODEL
  - Response: Proxy response as JSON (OpenAI-compatible embedding object)

- POST /index
  - Body:
    - id?: string   # optional client-provided id
    - text: string
    - metadata?: object
  - Response: { ok: true, id: "<id>" }

- POST /search
  - Body:
    - query: string
    - top_k?: number (default 5)
  - Behavior: MVP uses naive substring-scoring; future use vector DB kNN
  - Response:
    - results: [ { id, score, text, metadata } ]

- POST /chat
  - Body:
    - messages: [ { role: "user|assistant|system", content: string } ]
    - top_k?: number (docs to retrieve)
    - model?: string (override default)
  - Behavior: retrieve top_k docs, append them as system context, call upstream `/chat/completions` or `/chat` endpoint at model proxy with messages.
  - Response: proxy response (OpenAI chat completion) plus additional structured fields in top-level JSON as present.

- POST /apply
  - Body:
    - file_path: string
    - patch: string (unified diff or simple replacement spec)
    - dry_run?: boolean (default true)
  - Behavior: MVP returns simulated result; real application occurs in the Zed extension if it chooses to apply returned patch.
  - Response: { ok: true, applied: boolean, file: string, patch: string }

Notes:
- All requests should return HTTP 4xx/5xx on error with JSON error messages.
- MVP uses SSL_VERIFY env convention: DISABLE_SSL_VERIFY=true toggles verify=False when agent calls the proxy.

---

## 7. Data Models & Formats

- Document
  - id: string
  - text: string
  - metadata: object (path, language, repo, timestamp, etc.)

- ChatMessage
  - role: "system" | "user" | "assistant"
  - content: string

- ChatResponse
  - Mirror of upstream chat completion JSON; typically includes:
    - choices: [ { index, message: { role, content }, finish_reason } ]
    - usage: { prompt_tokens, completion_tokens, total_tokens } (if provided by upstream)

- memory.json (seed file to pass to assistant in follow-up interactions)
  - This JSON contains persistent memory items (user preferences, common config hints). See the `memory.json` sample provided in the repo baseline.

---

## 8. Environment Variables & Configuration

- LOCAL_PROXY_BASE: URL of the model proxy to hit (default: http://127.0.0.1:5000/v1)
- DISABLE_SSL_VERIFY: "true" to disable SSL verifies (DEFAULT false)
- EMBED_MODEL: embedding model name to request from proxy (default: text-embedding-3-small)
- CHAT_MODEL: default chat model to request from proxy (default: gpt-4o)
- PORT: HTTP port for the agent service (default: 7860)
- STORE_BACKEND: "memory" | "chroma" | "faiss" | "hnswlib" (MVP default: memory)

These can be provided in a local `.env` or as process env variables.

---

## 9. Baseline Files (scaffold placed in ./rag_agent)

The following baseline files will be present in the initial repo. They are intentionally small stubs to get the project running and to be expanded later.

- zed_agent_service.py — FastAPI MVP service implementing the API described above.
- requirements.txt — pinned dependencies for the MVP (fastapi, uvicorn, requests, pydantic).
- README.md — quick start for running the agent and connecting Zed.
- memory.json — starter memory object for the assistant (preferences & short notes).
- .env.example — environment variable examples for local runs.
- zed-extension/
  - extension.yaml — Zed extension manifest skeleton.
  - src/extension.ts — TypeScript skeleton that calls `/chat` and displays the assistant response and allow/deny modal.
- LICENSE, CONTRIBUTING.md — standard project metadata.

(These baseline files are included in the provided scaffold; use them as the initial commit and iterate.)

---

## 10. Security Considerations

- The service binds to localhost by default; do not expose it publicly unless behind an authenticated gateway.
- The agent will call an upstream model proxy; ensure credentials and tokens stay in your environment, not in committed files.
- Default behavior is to never auto-apply patches; user approval required in Zed for any workspace modifications.
- If you enable DISABLE_SSL_VERIFY=true while calling external upstreams, mark it clearly and only use it temporarily in trusted networks.

---

## 11. UX & Zed Integration

Zed extension responsibilities:
- Register a user command (example: "Zed RAG Agent: Ask/Refactor").
- Gather context: selected text, active file path, small surrounding lines (configurable), and repo metadata.
- Post to `http://127.0.0.1:7860/chat` with messages and `top_k`.
- Present assistant output in a modal including:
  - Assistant explanation / plan
  - Suggested patch preview (diff or inline)
  - Buttons: Allow / Deny / Copy to Clipboard
- On Allow: either apply the patch using Zed editing APIs or call `/apply` and then apply received patch locally.
- Keep interactions small (only send bounded context) to keep token usage low.

Notes on minimizing token usage:
- Send only selected code and small file context (e.g., ±30 lines) not entire files.
- Use embeddings/retrieval to provide relevant long-context material instead of sending whole files.
- Prefer simple summarization of long docs before inclusion in prompt.

---

## 12. Run/Dev Instructions (MVP)

1. Create a project directory and copy scaffold files (the repo root for the agent).
2. Create and activate a Python virtualenv:
   - python -m venv .venv
   - source .venv/bin/activate  (or .venv\Scripts\activate on Windows)
3. Install:
   - pip install -r requirements.txt
4. Run the agent service:
   - export LOCAL_PROXY_BASE=http://127.0.0.1:5000/v1
   - uvicorn zed_agent_service:app --reload --port 7860
5. In Zed:
   - Install the local extension (drop it into %LOCALAPPDATA%\Zed\extensions\your-ext).
   - Activate the command and test by asking sample prompts.

Testing tips:
- Use `curl` or Postman to call /index, /search, /chat endpoints before wiring Zed.
- Index a few docs locally using /index to validate retrieval behavior.

---

## 13. Acceptance Criteria

AC1. Agent service starts and responds to GET / with ok:true.  
AC2. /embed accepts an array of strings and returns an embedding response (proxy-forwarded).  
AC3. /index stores a document and returns a stable id.  
AC4. /search returns matches for indexed docs (MVP substring scoring).  
AC5. /chat successfully calls upstream model proxy and returns an assistant reply that references retrieved docs.  
AC6. Zed extension can call /chat and display response text; it must show a preview patch and require user Allow to apply changes.

---

## 14. Testing Plan

- Unit tests for:
  - Indexing and search behavior (document lifecycle)
  - Chat prompt assembly (ensures retrieved docs are appended)
  - Error handling when upstream proxy is unavailable
- Integration tests:
  - Start agent service with a mocked upstream proxy to validate /embed and /chat flows.
  - Simulate Zed extension requests with sample context and verify flow to /apply.
- Manual tests:
  - In a dev environment, index real code snippets and ensure the RAG prompt returns relevant suggestions while minimizing token sizes.

---

## 15. Roadmap (next milestones)

M1 (MVP) — Complete scaffold and Zed extension skeleton with in-memory store and proxy calls (this deliverable).  
M2 — Add vector store backend (Chroma + hnswlib) and an incremental indexer (filesystem watcher).  
M3 — Add a reranker (cross-encoder) to refine top-K results before sending to chat model.  
M4 — Add streaming responses (SSE) for progressive assistant outputs.  
M5 — Harden security, add local auth token, enable optional HTTPS, and containerize with Docker.  
M6 — Publish to a VCS repo, add CI (lint, unit tests, deployment steps), and create installable Zed package.

---

## 16. Appendices

A. memory.json (usage)
- Contains a compact JSON object with user preferences and common hints for subsequent agent sessions. Example structure:
  - preferences: { "editor": "Zed", "max_context_lines": 60, "preferred_models": { "chat": "gpt-4o", "embed": "text-embedding-3-small" } }
  - projects: [ { "name": "...", "notes": "..." } ]
- The sample `memory.json` is included in the repo scaffold.

B. Baseline file summary
- zed_agent_service.py — main app
- requirements.txt — dependencies
- README.md — quick setup
- .env.example — sample env variables
- memory.json — seed memory
- zed-extension/extension.yaml — manifest
- zed-extension/src/extension.ts — skeleton that calls /chat and shows the response

---

If you confirm this SRD, I will produce the memory.json seed and the baseline files in the `./rag_agent` directory. The SRD is intentionally implementation-agnostic so you can iterate quickly from the MVP to production-grade components.