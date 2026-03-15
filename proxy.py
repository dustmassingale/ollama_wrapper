"""
Ollama Proxy Server with GitHub Models + GitHub Copilot + AWS Bedrock support.

- Ollama API requests are proxied to the remote server at REMOTE_OLLAMA_URL.
- GH | models use the azure-ai-inference SDK against
  https://models.inference.ai.azure.com with a standard GitHub PAT.
- GC | models use the GitHub Copilot Chat API (api.githubcopilot.com) via
  an OAuth token obtained through the GitHub device flow. Run:
    GET http://localhost:5000/auth/copilot
  and follow the instructions to log in through your browser.
- BR | models use the AWS Bedrock converse/invoke API via boto3.
  Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_BEDROCK_REGION in .env.
  Only models NOT already covered by GH | or GC | are advertised (fill-in-
  the-gap strategy: prefer free GitHub quota before pay-per-use Bedrock).
- Responses are translated into Ollama-format NDJSON so any Ollama client
  (Continue.dev, Open WebUI, etc.) works transparently.
"""

import json
import logging
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import requests
import urllib3
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context

_boto3_available: bool
try:
    import boto3 as boto3  # noqa: PLC0414

    _boto3_available = True
except ImportError:
    _boto3_available = False

# Public alias used throughout the module
_BOTO3_AVAILABLE = _boto3_available

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

# ---------------------------------------------------------------------------
# SSL verification
# ---------------------------------------------------------------------------
# DISABLE_SSL_VERIFY=true bypasses SSL certificate validation.
# ⚠️  WARNING: Do NOT enable in a production environment.
#     This is intended as a temporary workaround for corporate environments
#     that use self-signed certificates in their certificate chain.
#     Only use for testing or while a proper certificate fix is arranged.
SSL_VERIFY: bool = os.getenv("DISABLE_SSL_VERIFY", "false").strip().lower() != "true"

if not SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    log.warning(
        "⚠️  SSL verification is DISABLED (DISABLE_SSL_VERIFY=true). "
        "Do NOT use this setting in a production environment."
    )

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

# ---------------------------------------------------------------------------
# AWS Bedrock configuration
# ---------------------------------------------------------------------------

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-east-1")

BR_PREFIX = "BR | "

# Canonical Bedrock model IDs we want to advertise (fill-in-the-gap catalogue).
# These are pruned at startup against the live GH | + GC | catalogue so we
# never duplicate a model already available for free via GitHub.
#
# All entries have been verified working with the CodeLight IAM user.
# Models that require on-demand throughput use inference profile IDs (us.* prefix)
# rather than bare model IDs — Bedrock requires this for most non-Mistral models.
_BEDROCK_CANDIDATE_MODELS: list[dict[str, str]] = [
    # --- Anthropic Claude (inference profiles — verified OK) ---
    {
        "sdk_name": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude 3 Haiku",
    },
    {
        "sdk_name": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude 3.5 Haiku",
    },
    {
        "sdk_name": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude Haiku 4.5",
    },
    {
        "sdk_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude Sonnet 4",
    },
    {
        "sdk_name": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude Sonnet 4.5",
    },
    {
        "sdk_name": "us.anthropic.claude-sonnet-4-6",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude Sonnet 4.6",
    },
    {
        "sdk_name": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude Opus 4.5",
    },
    {
        "sdk_name": "us.anthropic.claude-opus-4-6-v1",
        "family": "claude",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Claude Opus 4.6",
    },
    # --- Mistral (direct model IDs — verified OK, no inference profile needed) ---
    {
        "sdk_name": "mistral.mistral-large-3-675b-instruct",
        "family": "mistral",
        "parameter_size": "675B",
        "kind": "chat",
        "display_hint": "Mistral Large 3",
    },
    {
        "sdk_name": "mistral.ministral-3-14b-instruct",
        "family": "mistral",
        "parameter_size": "14B",
        "kind": "chat",
        "display_hint": "Ministral 14B 3.0",
    },
    {
        "sdk_name": "mistral.ministral-3-8b-instruct",
        "family": "mistral",
        "parameter_size": "8B",
        "kind": "chat",
        "display_hint": "Ministral 3 8B",
    },
    {
        "sdk_name": "mistral.mistral-7b-instruct-v0:2",
        "family": "mistral",
        "parameter_size": "7B",
        "kind": "chat",
        "display_hint": "Mistral 7B Instruct",
    },
    {
        "sdk_name": "mistral.mixtral-8x7b-instruct-v0:1",
        "family": "mistral",
        "parameter_size": "8x7B",
        "kind": "chat",
        "display_hint": "Mixtral 8x7B",
    },
    {
        "sdk_name": "mistral.devstral-2-123b",
        "family": "mistral",
        "parameter_size": "123B",
        "kind": "chat",
        "display_hint": "Devstral 2 123B",
    },
    {
        "sdk_name": "mistral.magistral-small-2509",
        "family": "mistral",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Magistral Small",
    },
    # --- Meta Llama (inference profiles — verified OK) ---
    {
        "sdk_name": "us.meta.llama3-1-8b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "8B",
        "kind": "chat",
        "display_hint": "Llama 3.1 8B",
    },
    {
        "sdk_name": "us.meta.llama3-1-70b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "70B",
        "kind": "chat",
        "display_hint": "Llama 3.1 70B",
    },
    {
        "sdk_name": "us.meta.llama3-2-1b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "1B",
        "kind": "chat",
        "display_hint": "Llama 3.2 1B",
    },
    {
        "sdk_name": "us.meta.llama3-2-3b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "3B",
        "kind": "chat",
        "display_hint": "Llama 3.2 3B",
    },
    {
        "sdk_name": "us.meta.llama3-2-11b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "11B",
        "kind": "chat",
        "display_hint": "Llama 3.2 11B",
    },
    {
        "sdk_name": "us.meta.llama3-2-90b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "90B",
        "kind": "chat",
        "display_hint": "Llama 3.2 90B",
    },
    {
        "sdk_name": "us.meta.llama3-3-70b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "70B",
        "kind": "chat",
        "display_hint": "Llama 3.3 70B",
    },
    {
        "sdk_name": "us.meta.llama4-scout-17b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "17B",
        "kind": "chat",
        "display_hint": "Llama 4 Scout 17B",
    },
    {
        "sdk_name": "us.meta.llama4-maverick-17b-instruct-v1:0",
        "family": "llama",
        "parameter_size": "17B",
        "kind": "chat",
        "display_hint": "Llama 4 Maverick 17B",
    },
    # --- Amazon Nova (inference profiles — verified OK) ---
    {
        "sdk_name": "us.amazon.nova-micro-v1:0",
        "family": "nova",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Nova Micro",
    },
    {
        "sdk_name": "us.amazon.nova-lite-v1:0",
        "family": "nova",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Nova Lite",
    },
    {
        "sdk_name": "us.amazon.nova-pro-v1:0",
        "family": "nova",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Nova Pro",
    },
    {
        "sdk_name": "us.amazon.nova-premier-v1:0",
        "family": "nova",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Nova Premier",
    },
    {
        "sdk_name": "us.amazon.nova-2-lite-v1:0",
        "family": "nova",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Nova 2 Lite",
    },
    # --- DeepSeek (inference profile — verified OK) ---
    {
        "sdk_name": "us.deepseek.r1-v1:0",
        "family": "deepseek",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "DeepSeek R1",
    },
    # --- Writer Palmyra (inference profiles — verified OK) ---
    {
        "sdk_name": "us.writer.palmyra-x4-v1:0",
        "family": "palmyra",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Palmyra X4",
    },
    {
        "sdk_name": "us.writer.palmyra-x5-v1:0",
        "family": "palmyra",
        "parameter_size": "unknown",
        "kind": "chat",
        "display_hint": "Palmyra X5",
    },
    # --- OpenAI on Bedrock (direct model ID — verified OK) ---
    {
        "sdk_name": "openai.gpt-oss-120b-1:0",
        "family": "gpt",
        "parameter_size": "120B",
        "kind": "chat",
        "display_hint": "GPT OSS 120B",
    },
    # --- Embeddings ---
    # Note: us.cohere.embed-v4:0 is GATE (needs console enable) — omitted for now.
    # Titan v1 and v2 verified OK with direct model IDs (invoke_model API).
    {
        "sdk_name": "amazon.titan-embed-text-v1",
        "family": "titan",
        "parameter_size": "unknown",
        "kind": "embed",
        "display_hint": "Titan Embed Text v1",
    },
    {
        "sdk_name": "amazon.titan-embed-text-v2:0",
        "family": "titan",
        "parameter_size": "unknown",
        "kind": "embed",
        "display_hint": "Titan Embed Text v2",
    },
]

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
_pending_device_flow: dict[str, object] = {}


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


def _discover_copilot_models() -> list[dict[str, object]]:
    """Query api.githubcopilot.com/models, probe each, and return working chat models."""
    token = _get_copilot_token()
    if not token:
        return []
    try:
        resp = requests.get(
            f"{COPILOT_CHAT_ENDPOINT}/models",
            headers={**COPILOT_CHAT_HEADERS, "Authorization": f"Bearer {token}"},
            timeout=10,
            verify=SSL_VERIFY,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])

        # The Copilot /models endpoint exposes capabilities.type for each model.
        # Use it directly; fall back to name-based guessing only if absent.
        # Currently the Copilot API only exposes chat models, but guard for embed
        # entries appearing in future by checking capabilities.type.
        chat_candidates = []
        embed_candidates = []
        for m in raw:
            mid = m.get("id", "")
            if not mid:
                continue
            cap_type = m.get("capabilities", {}).get("type", "")
            if cap_type == "embeddings":
                embed_candidates.append(mid)
            elif cap_type == "chat" or cap_type == "":
                # Unknown cap_type treated as chat (safe default for Copilot)
                chat_candidates.append(mid)
            # Anything else (image, tts, …) is silently skipped

        log.info(
            "Probing %d chat + %d embed Copilot candidates...",
            len(chat_candidates),
            len(embed_candidates),
        )
        entries = []

        for mid in chat_candidates:
            if _probe_chat_model(
                token, mid, COPILOT_CHAT_ENDPOINT, COPILOT_CHAT_HEADERS
            ):
                log.debug("  OK (chat) %s", mid)
                entries.append(
                    {
                        "name": f"{GC_PREFIX}{mid}",
                        "sdk_name": mid,
                        "token": "copilot",
                        "kind": "chat",
                        "size": 0,
                        "modified_at": "2024-01-01T00:00:00Z",
                        "details": {
                            "family": mid.split("-")[0].lower(),
                            "parameter_size": "unknown",
                        },
                    }
                )
            else:
                log.debug("  -- (chat) %s (skipped — probe failed)", mid)

        for mid in embed_candidates:
            if _probe_embed_model(
                token, mid, COPILOT_CHAT_ENDPOINT, COPILOT_CHAT_HEADERS
            ):
                log.debug("  OK (embed) %s", mid)
                entries.append(
                    {
                        "name": f"{GC_PREFIX}{mid}",
                        "sdk_name": mid,
                        "token": "copilot",
                        "kind": "embed",
                        "size": 0,
                        "modified_at": "2024-01-01T00:00:00Z",
                        "details": {
                            "family": mid.split("-")[0].lower(),
                            "parameter_size": "unknown",
                        },
                    }
                )
            else:
                log.debug("  -- (embed) %s (skipped — probe failed)", mid)

        chat_count = sum(1 for e in entries if e.get("kind") == "chat")
        embed_count = sum(1 for e in entries if e.get("kind") == "embed")
        log.info(
            "Discovered %d chat + %d embed Copilot models (GC |)",
            chat_count,
            embed_count,
        )
        return entries
    except Exception as e:
        log.warning("Copilot model discovery failed: %s", e)
        return []


def _rebuild_catalogue(
    existing_gh_models: list[dict[str, object]] | None = None,
) -> None:
    """Rebuild GITHUB_MODELS + GITHUB_MODEL_MAP after a new Copilot token arrives.

    Pass ``existing_gh_models`` to reuse an already-discovered GH list and
    avoid a redundant second probe run at startup.
    """
    global GITHUB_MODELS, GITHUB_MODEL_MAP  # noqa: PLW0603
    gh_models = (
        existing_gh_models
        if existing_gh_models is not None
        else _discover_models(GITHUB_TOKEN, GH_PREFIX, "standard")
    )
    gc_models = _discover_copilot_models()
    # Keep GC models whose sdk_name doesn't already appear under GH | so we
    # don't advertise e.g. both "GH | gpt-4o" and "GC | gpt-4o".
    gh_sdk_names = {m["sdk_name"] for m in gh_models}
    unique_gc = [m for m in gc_models if m["sdk_name"] not in gh_sdk_names]

    # Bedrock: fill-in-the-gap — only include models not already in GH or GC.
    covered_sdk_names = gh_sdk_names | {m["sdk_name"] for m in unique_gc}
    br_models = _build_bedrock_catalogue(covered_sdk_names)

    GITHUB_MODELS = gh_models + unique_gc + br_models  # type: ignore[assignment,operator]
    GITHUB_MODEL_MAP = {str(m["name"]): m for m in GITHUB_MODELS}  # type: ignore[assignment]
    log.info(
        "Catalogue rebuilt: %d GH | + %d GC | + %d BR | = %d total",
        len(gh_models),
        len(unique_gc),
        len(br_models),
        len(GITHUB_MODELS),
    )


# ---------------------------------------------------------------------------
# AWS Bedrock catalogue builder
# ---------------------------------------------------------------------------


def _bedrock_client():  # type: ignore[return]
    """Return a boto3 bedrock-runtime client, or None if credentials are missing."""
    if not _boto3_available:
        log.warning(
            "boto3 is not installed — BR | models unavailable. Run: pip install boto3"
        )
        return None
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        return None
    return boto3.client(  # type: ignore[possibly-unbound,union-attr]
        "bedrock-runtime",
        region_name=AWS_BEDROCK_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def _build_bedrock_catalogue(
    already_covered_sdk_names: set[str],
) -> list[dict[str, object]]:
    """
    Build the BR | model list from the static candidate list, excluding any
    sdk_name that is already covered by GH | or GC | models.

    We do NOT probe Bedrock models at startup (each call costs money).
    Instead we include all candidates not already covered and let call-time
    errors surface naturally.
    """
    if not _boto3_available:
        return []
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        log.info("AWS credentials not configured — BR | models unavailable")
        return []

    # Normalise covered names: strip dots/dashes for fuzzy dedup.
    # e.g. "Meta-Llama-3.1-8B-Instruct" (GH) vs "meta.llama3-1-8b-instruct-v1:0" (BR)
    # are different enough that exact dedup is fine here — we just don't want
    # an exact sdk_name collision.
    entries = []
    for c in _BEDROCK_CANDIDATE_MODELS:
        if c["sdk_name"] in already_covered_sdk_names:
            log.debug("BR | skip %s — already covered by GH/GC", c["sdk_name"])
            continue
        entry = {
            "name": f"{BR_PREFIX}{c['sdk_name']}",
            "sdk_name": c["sdk_name"],
            "token": "bedrock",
            "kind": c["kind"],
            "size": 0,
            "modified_at": "2024-01-01T00:00:00Z",
            "details": {
                "family": c["family"],
                "parameter_size": c["parameter_size"],
            },
        }
        entries.append(entry)
        log.debug("BR | +  %s  (%s)", c["sdk_name"], c["kind"])

    log.info(
        "Bedrock catalogue: %d chat + %d embed models (BR |)",
        sum(1 for e in entries if e["kind"] == "chat"),
        sum(1 for e in entries if e["kind"] == "embed"),
    )
    return entries


# ---------------------------------------------------------------------------
# AWS Bedrock chat helpers
# ---------------------------------------------------------------------------


def _bedrock_messages_to_converse(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    """
    Convert Ollama-style messages into the format expected by Bedrock converse().
    Returns (messages_list, system_prompt_string).
    The Bedrock converse API keeps system prompts separate from the message list.
    """
    system_prompt = ""
    converse_messages: list[dict[str, object]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        if role == "system":
            system_prompt = content
        else:
            converse_messages.append(
                {
                    "role": "user" if role == "user" else "assistant",
                    "content": [{"text": content}],
                }
            )
    return converse_messages, system_prompt


def _bedrock_chat_streaming(model_name: str, messages: list[dict[str, object]]):
    """
    Call Bedrock converse_stream() and yield Ollama-format NDJSON lines.
    """
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(model_name), {})
    sdk_name = entry.get(
        "sdk_name",
        model_name[len(BR_PREFIX) :]
        if model_name.startswith(BR_PREFIX)
        else model_name,
    )

    client = _bedrock_client()
    if client is None:
        yield (
            json.dumps(
                {
                    "model": model_name,
                    "created_at": now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "error",
                    "error": "AWS credentials not configured.",
                }
            )
            + "\n"
        )
        return

    converse_msgs, system_prompt = _bedrock_messages_to_converse(messages)
    kwargs: dict[str, object] = {"modelId": sdk_name, "messages": converse_msgs}
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    try:
        response = client.converse_stream(**kwargs)
        stream = response.get("stream")
        if not stream:
            raise RuntimeError("No stream in Bedrock response")

        for event in stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield (
                        json.dumps(
                            {
                                "model": model_name,
                                "created_at": now_iso(),
                                "message": {"role": "assistant", "content": text},
                                "done": False,
                            }
                        )
                        + "\n"
                    )
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "stop")
                yield (
                    json.dumps(
                        {
                            "model": model_name,
                            "created_at": now_iso(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "done_reason": stop_reason,
                            "total_duration": 0,
                            "load_duration": 0,
                            "prompt_eval_count": 0,
                            "eval_count": 0,
                        }
                    )
                    + "\n"
                )
                return

        # Fallback done if stream ends without messageStop
        yield (
            json.dumps(
                {
                    "model": model_name,
                    "created_at": now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                }
            )
            + "\n"
        )

    except Exception as e:
        log.error("Bedrock streaming error for %s: %s", sdk_name, e)
        yield (
            json.dumps(
                {
                    "model": model_name,
                    "created_at": now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "error",
                    "error": str(e),
                }
            )
            + "\n"
        )


def _bedrock_chat_blocking(
    model_name: str, messages: list[dict[str, object]]
) -> dict[str, object]:
    """
    Call Bedrock converse() and return a single Ollama-format response dict.
    """
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(model_name), {})
    sdk_name = entry.get(
        "sdk_name",
        model_name[len(BR_PREFIX) :]
        if model_name.startswith(BR_PREFIX)
        else model_name,
    )

    client = _bedrock_client()
    if client is None:
        return {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "error",
            "error": "AWS credentials not configured.",
        }

    converse_msgs, system_prompt = _bedrock_messages_to_converse(messages)
    kwargs: dict[str, object] = {"modelId": sdk_name, "messages": converse_msgs}
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    try:
        response = client.converse(**kwargs)
        output = response.get("output", {})
        content_blocks = output.get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks)
        usage = response.get("usage", {})
        return {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": usage.get("inputTokens", 0),
            "eval_count": usage.get("outputTokens", 0),
        }
    except Exception as e:
        log.error("Bedrock blocking error for %s: %s", sdk_name, e)
        return {
            "model": model_name,
            "created_at": now_iso(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "error",
            "error": str(e),
        }


def _bedrock_embed(model_name: str, input_text: str) -> dict[str, object]:
    """
    Call Bedrock invoke_model() for Titan embedding models and return a
    dict with an 'embedding' key (list of floats).
    """
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(model_name), {})
    sdk_name = entry.get(
        "sdk_name",
        model_name[len(BR_PREFIX) :]
        if model_name.startswith(BR_PREFIX)
        else model_name,
    )

    client = _bedrock_client()
    if client is None:
        return {"error": "AWS credentials not configured."}

    try:
        body = json.dumps({"inputText": input_text})
        response = client.invoke_model(
            modelId=sdk_name,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return {"embedding": result.get("embedding", [])}
    except Exception as e:
        log.error("Bedrock embed error for %s: %s", sdk_name, e)
        return {"error": str(e)}


def _bedrock_chat_streaming_openai(model_name: str, messages: list[dict[str, object]]):
    """
    Stream a Bedrock response as OpenAI-compat SSE (for /v1/chat/completions).
    """
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(model_name), {})
    sdk_name = entry.get(
        "sdk_name",
        model_name[len(BR_PREFIX) :]
        if model_name.startswith(BR_PREFIX)
        else model_name,
    )

    client = _bedrock_client()
    if client is None:
        err = {
            "error": {"message": "AWS credentials not configured.", "type": "api_error"}
        }
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
        return

    converse_msgs, system_prompt = _bedrock_messages_to_converse(messages)
    kwargs: dict[str, object] = {"modelId": sdk_name, "messages": converse_msgs}
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    created = int(datetime.now(timezone.utc).timestamp())

    try:
        response = client.converse_stream(**kwargs)
        stream = response.get("stream")
        if not stream:
            raise RuntimeError("No stream in Bedrock response")

        for event in stream:
            if "contentBlockDelta" in event:
                text = event["contentBlockDelta"].get("delta", {}).get("text", "")
                if text:
                    chunk = {
                        "id": "chatcmpl-bedrock",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "stop")
                chunk = {
                    "id": "chatcmpl-bedrock",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": stop_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        log.error("Bedrock SSE streaming error for %s: %s", sdk_name, e)
        err = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


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


def _probe_chat_model(
    token: str,
    sdk_name: str,
    endpoint: str,
    extra_headers: dict[str, str] | None = None,
) -> bool:
    """Return True if the model responds 200 to a minimal chat request."""
    try:
        r = requests.post(
            f"{endpoint}/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                **(extra_headers or {}),
            },
            json={
                "model": sdk_name,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=15,
            verify=SSL_VERIFY,
        )
        return r.status_code == 200
    except Exception:
        return False


def _probe_embed_model(
    token: str,
    sdk_name: str,
    endpoint: str,
    extra_headers: dict[str, str] | None = None,
) -> bool:
    """Return True if the model responds 200 to a minimal embeddings request.

    Cohere models require ``input`` as a list and a valid ``input_type``.
    OpenAI-style models accept a plain string.  We try the Cohere shape first
    (it is a strict superset), so one probe covers both families.
    """
    try:
        r = requests.post(
            f"{endpoint}/embeddings",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                **(extra_headers or {}),
            },
            json={
                "model": sdk_name,
                "input": ["hello"],
                "input_type": "text",
            },
            timeout=15,
            verify=SSL_VERIFY,
        )
        return r.status_code == 200
    except Exception:
        return False


# Dimensions reported by each known GH embed model (informational only).
_GH_EMBED_DIMS: dict[str, int] = {
    "Cohere-embed-v3-english": 1024,
    "Cohere-embed-v3-multilingual": 1024,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}

# Cohere embed models need input_type; OpenAI-style ones don't.
_COHERE_EMBED_FAMILY = {"Cohere-embed-v3-english", "Cohere-embed-v3-multilingual"}


def _discover_models(
    token: str, prefix: str, token_label: str
) -> list[dict[str, object]]:
    """
    Query the GitHub Models catalogue endpoint and return all working models,
    split by type:
      - ``model_type == "chat-completion"``  → probe with /chat/completions
      - ``model_type == "embeddings"``        → probe with /embeddings
      - anything else (image, tts, …)         → skipped

    Falls back to the hardcoded chat-only list on any error.
    """
    if not token:
        return []

    try:
        resp = requests.get(
            f"{GITHUB_INFERENCE_ENDPOINT}/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
            verify=SSL_VERIFY,
        )
        resp.raise_for_status()
        raw_models = resp.json()

        # Partition by the "task" field reported by the API — no name-guessing needed.
        # The GitHub Models /models endpoint returns task="chat-completion" or
        # task="embeddings" for each entry.
        chat_candidates = []
        embed_candidates = []
        for m in raw_models:
            name = m.get("name", "")
            task = m.get("task", "")
            if not name:
                continue
            if task == "chat-completion":
                chat_candidates.append(name)
            elif task == "embeddings":
                embed_candidates.append(name)
            else:
                log.debug("GH skip %s (task=%r)", name, task)

        log.info(
            "Probing %d chat + %d embed candidates for %s token...",
            len(chat_candidates),
            len(embed_candidates),
            token_label,
        )

        entries = []

        for sdk_name in chat_candidates:
            if _probe_chat_model(token, sdk_name, GITHUB_INFERENCE_ENDPOINT):
                log.debug("  OK (chat) %s", sdk_name)
                entries.append(
                    {
                        "name": f"{prefix}{sdk_name}",
                        "sdk_name": sdk_name,
                        "token": token_label,
                        "kind": "chat",
                        "size": 0,
                        "modified_at": "2024-01-01T00:00:00Z",
                        "details": {
                            "family": sdk_name.split("-")[0].lower(),
                            "parameter_size": "unknown",
                        },
                    }
                )
            else:
                log.debug("  -- (chat) %s (skipped — probe failed)", sdk_name)

        for sdk_name in embed_candidates:
            if _probe_embed_model(token, sdk_name, GITHUB_INFERENCE_ENDPOINT):
                log.debug("  OK (embed) %s", sdk_name)
                entries.append(
                    {
                        "name": f"{prefix}{sdk_name}",
                        "sdk_name": sdk_name,
                        "token": token_label,
                        "kind": "embed",
                        "size": 0,
                        "modified_at": "2024-01-01T00:00:00Z",
                        "details": {
                            "family": sdk_name.split("-")[0].lower(),
                            "parameter_size": "unknown",
                            "dimensions": _GH_EMBED_DIMS.get(sdk_name, 0),
                        },
                    }
                )
            else:
                log.debug("  -- (embed) %s (skipped — probe failed)", sdk_name)

        chat_count = sum(1 for e in entries if e.get("kind") == "chat")
        embed_count = sum(1 for e in entries if e.get("kind") == "embed")
        log.info(
            "Discovered %d chat + %d embed models for %s token (prefix '%s')",
            chat_count,
            embed_count,
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
                "kind": "chat",
                "size": 0,
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {"family": f["family"], "parameter_size": f["size"]},
            }
            for f in _FALLBACK_MODELS
        ]


# ---------------------------------------------------------------------------
# GitHub model catalogue — built at startup via live API discovery
# ---------------------------------------------------------------------------

GH_PREFIX = "GH | "
GC_PREFIX = "GC | "
REMOTE_PREFIX = "155 | "
# BR_PREFIX is defined earlier alongside Bedrock config

# Populated at module load time; rebuilt after Copilot login via _rebuild_catalogue().
# Discover GH | models once; pass them into _rebuild_catalogue so it doesn't
# repeat the same probe run a second time when a Copilot token is present.
_initial_gh_models: list[dict[str, object]] = _discover_models(
    GITHUB_TOKEN, GH_PREFIX, "standard"
)
# GITHUB_MODELS and GITHUB_MODEL_MAP are always assigned below (one of the two branches).
GITHUB_MODELS: list[dict[str, object]] = []
GITHUB_MODEL_MAP: dict[str, dict[str, object]] = {}

if _get_copilot_token():
    # Copilot token already in .env — build the full catalogue in one shot,
    # reusing the GH | probe results we just obtained.
    _rebuild_catalogue(existing_gh_models=_initial_gh_models)
else:
    # No Copilot token yet — start with GH | + BR | models.
    _gh_sdk_names: set[str] = {str(m["sdk_name"]) for m in _initial_gh_models}
    _br_models = _build_bedrock_catalogue(_gh_sdk_names)
    GITHUB_MODELS = _initial_gh_models + _br_models  # type: ignore[assignment,operator]
    GITHUB_MODEL_MAP = {str(m["name"]): m for m in GITHUB_MODELS}  # type: ignore[assignment]


def _active_github_models() -> list[dict[str, object]]:
    """Return all currently known models (GH |, GC |, BR |)."""
    return GITHUB_MODELS


def is_bedrock_model(name: str) -> bool:
    """Return True if this is a BR | chat model in the current catalogue."""
    canonical = _canonical_gh_name(name)
    entry = GITHUB_MODEL_MAP.get(canonical, {})
    return entry.get("token") == "bedrock" and entry.get("kind") == "chat"


def is_bedrock_embed_model(name: str) -> bool:
    """Return True if this is a BR | embedding model."""
    canonical = _canonical_gh_name(name)
    entry = GITHUB_MODEL_MAP.get(canonical, {})
    return entry.get("token") == "bedrock" and entry.get("kind") == "embed"


def is_any_embed_model(name: str) -> bool:
    """Return True if this model (GH |, GC |, or BR |) is an embedding model."""
    canonical = _canonical_gh_name(name)
    entry = GITHUB_MODEL_MAP.get(canonical, {})
    return entry.get("kind") == "embed"


def _token_for_model(name: str) -> str:
    """Return the correct bearer token for the given canonical model name.
    Raises RuntimeError for Bedrock models (they use boto3, not a bearer token).
    """
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(name), {})
    if entry.get("token") == "bedrock":
        raise RuntimeError(
            f"'{name}' is a Bedrock model — use _bedrock_chat_* helpers directly."
        )
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
    """Return True for GH | and GC | chat models only.

    Excludes:
      - BR | Bedrock models (token == "bedrock")
      - Any embed-kind model (kind == "embed") — those are handled by
        _handle_embed_request, not the chat routes.
    """
    canonical = _canonical_gh_name(name)
    entry = GITHUB_MODEL_MAP.get(canonical, {})
    return (
        canonical in GITHUB_MODEL_MAP
        and entry.get("token") != "bedrock"
        and entry.get("kind") != "embed"
    )


def _canonical_gh_name(name: str) -> str:
    """Return the canonical prefixed display name regardless of how the client sent it.

    Handles:
      - "GH | gpt-4o"                        (already canonical, standard)
      - "GC | claude-haiku-4.5"              (already canonical, copilot)
      - "BR | anthropic.claude-3-haiku-..."  (already canonical, bedrock)
      - "gpt-4o"                             (bare — tried with all prefixes)
      - "claude-haiku-4.5"                   (bare with dots — normalised)
    """
    if name in GITHUB_MODEL_MAP:
        return name
    for prefix in (GH_PREFIX, GC_PREFIX, BR_PREFIX):
        prefixed = prefix + name
        if prefixed in GITHUB_MODEL_MAP:
            return prefixed
    # Normalise dots to dashes and retry with all prefixes
    normalised = name.replace(".", "-")
    if normalised in GITHUB_MODEL_MAP:
        return normalised
    for prefix in (GH_PREFIX, GC_PREFIX, BR_PREFIX):
        prefixed_normalised = prefix + normalised
        if prefixed_normalised in GITHUB_MODEL_MAP:
            return prefixed_normalised
    return name


def get_sdk_name(name: str) -> str:
    """Return the backend model ID for a display name.

    Resolution order:
      1. Look up the canonical prefixed name in GITHUB_MODEL_MAP.
      2. If not found, strip any known prefix and return the bare name so the
         backend receives a usable model ID rather than a display string like
         "GH | Mistral-Nemo" or "BR | anthropic.claude-3-haiku-20240307".
    """
    canonical = _canonical_gh_name(name)
    entry = GITHUB_MODEL_MAP.get(canonical)
    if entry:
        sdk = str(entry["sdk_name"])
        log.debug("get_sdk_name(%r) -> canonical=%r sdk_name=%r", name, canonical, sdk)
        return sdk
    # Fallback: strip prefix so we at least send the bare model name
    for prefix in (GH_PREFIX, GC_PREFIX, BR_PREFIX, REMOTE_PREFIX):
        if name.startswith(prefix):
            bare = name[len(prefix) :]
            log.warning(
                "get_sdk_name(%r): not found in GITHUB_MODEL_MAP "
                "(canonical=%r), falling back to bare name %r. Known keys: %s",
                name,
                canonical,
                bare,
                list(GITHUB_MODEL_MAP.keys()),
            )
            return bare
    log.warning(
        "get_sdk_name(%r): not found in GITHUB_MODEL_MAP and no prefix "
        "matched, returning as-is. Known keys: %s",
        name,
        list(GITHUB_MODEL_MAP.keys()),
    )
    return name


# ---------------------------------------------------------------------------
# Global Flask error handlers — always JSON, never HTML
# ---------------------------------------------------------------------------


@app.errorhandler(404)
def not_found(e: Exception):
    log.warning("404 %s %s", request.method, request.path)
    return jsonify({"error": "not found", "path": request.path}), 404


@app.errorhandler(405)
def method_not_allowed(e: Exception):
    log.warning("405 %s %s", request.method, request.path)
    return jsonify({"error": "method not allowed"}), 405


@app.errorhandler(Exception)
def unhandled(e: Exception):
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
        verify=SSL_VERIFY,
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
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(model_name), {})
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


def _ollama_messages_to_sdk(messages: list[dict[str, object]]):
    """Convert Ollama-style message dicts to azure-ai-inference message objects."""
    result = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        if role == "system":
            result.append(SystemMessage(content))
        elif role == "assistant":
            result.append(AssistantMessage(content))
        else:
            result.append(UserMessage(content))
    return result


def _github_chat_streaming(model_name: str, messages: list[dict[str, object]]):
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
                chunk["total_duration"] = "0"
                chunk["load_duration"] = "0"
                chunk["prompt_eval_count"] = "0"
                chunk["eval_count"] = "0"

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


def _github_chat_blocking(
    model_name: str, messages: list[dict[str, object]]
) -> dict[str, object]:
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


def _github_generate_blocking(
    model_name: str, prompt: str, system: str = ""
) -> dict[str, object]:
    """
    Wrap a plain /api/generate prompt as a chat call to GitHub Models
    and return an Ollama /api/generate-format response dict.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    chat_result: dict[str, object] = _github_chat_blocking(model_name, messages)

    chat_msg2 = cast("dict[str, object]", chat_result["message"])
    return {
        "model": model_name,
        "created_at": chat_result["created_at"],
        "response": chat_msg2["content"],
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
    """Return model metadata. GitHub/Bedrock models get a synthetic response."""
    data = request.json or {}
    model_name = data.get("model") or data.get("name", "")
    log.debug("/api/show model=%r", model_name)

    canonical = _canonical_gh_name(model_name)
    entry = GITHUB_MODEL_MAP.get(canonical)

    if entry is not None:
        token_kind = entry.get("token", "github")
        kind = entry.get("kind", "chat")
        source = (
            "AWS Bedrock"
            if token_kind == "bedrock"
            else "GitHub Copilot"
            if token_kind == "copilot"
            else "GitHub Models"
        )
        entry_details = cast("dict[str, object]", entry["details"])
        details = {
            "format": "api",
            "family": entry_details.get("family", token_kind),
            "families": [entry_details.get("family", token_kind)],
            "parameter_size": entry_details.get("parameter_size", "unknown"),
            "quantization_level": "none",
        }
        if kind == "embed":
            dims = entry_details.get("dimensions", 0)
            if dims:
                details["dimensions"] = dims
        return jsonify(
            {
                "license": "",
                "modelfile": f"# {source} ({kind}): {model_name}",
                "parameters": "",
                "template": "{{ .Prompt }}" if kind == "chat" else "",
                "details": details,
                "model_info": {"kind": kind},
            }
        )

    # Strip "155 | " prefix before forwarding to remote Ollama
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
        resp = proxy_request("POST", "/api/show", data=data)
        return make_proxy_response(resp)

    # Unknown model — return a clean 404
    log.warning("/api/show unknown model %r", model_name)
    return jsonify({"error": f"Model '{model_name}' not found."}), 404


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

    if is_any_embed_model(model_name):
        return jsonify(
            {
                "error": (
                    f"'{model_name}' is an embedding model. "
                    "Use /api/embeddings or /api/embed instead."
                )
            }
        ), 400

    if is_bedrock_model(model_name):
        model_name = _canonical_gh_name(model_name)
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            return jsonify(
                {"error": "AWS credentials not configured on the proxy server."}
            ), 500
        if do_stream:
            return Response(
                stream_with_context(_bedrock_chat_streaming(model_name, messages)),
                status=200,
                content_type="application/x-ndjson",
            )
        else:
            result = _bedrock_chat_blocking(model_name, messages)
            if "error" in result:
                return jsonify(result), 502
            return jsonify(result)

    if is_github_model(model_name):
        model_name = _canonical_gh_name(model_name)
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

    if is_any_embed_model(model_name):
        return jsonify(
            {
                "error": (
                    f"'{model_name}' is an embedding model. "
                    "Use /api/embeddings or /api/embed instead."
                )
            }
        ), 400

    if is_bedrock_model(model_name):
        model_name = _canonical_gh_name(model_name)
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            return jsonify(
                {"error": "AWS credentials not configured on the proxy server."}
            ), 500
        # Wrap generate as a chat call (same approach as GitHub models)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if do_stream:
            return Response(
                stream_with_context(_bedrock_chat_streaming(model_name, messages)),
                status=200,
                content_type="application/x-ndjson",
            )
        else:
            chat_result = _bedrock_chat_blocking(model_name, messages)
            if "error" in chat_result:
                return jsonify(chat_result), 502
            # Convert chat response to generate format
            chat_msg = cast("dict[str, object]", chat_result["message"])
            return jsonify(
                {
                    "model": model_name,
                    "created_at": chat_result["created_at"],
                    "response": chat_msg["content"],
                    "done": True,
                    "done_reason": chat_result.get("done_reason", "stop"),
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": chat_result.get("prompt_eval_count", 0),
                    "eval_count": chat_result.get("eval_count", 0),
                }
            )

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
                    "total_tokens": cast(int, result.get("prompt_eval_count", 0))
                    + cast(int, result.get("eval_count", 0)),
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


def _is_gh_embed_model(name: str) -> bool:
    """Return True if this is a GH | or GC | embedding model."""
    canonical = _canonical_gh_name(name)
    entry = GITHUB_MODEL_MAP.get(canonical, {})
    return entry.get("token") not in ("bedrock", "") and entry.get("kind") == "embed"


def _gh_embed(model_name: str, inputs: list[str]) -> dict[str, object]:
    """
    Call the GitHub Models (Azure AI Inference) embeddings endpoint.

    Handles two families:
      - Cohere  (Cohere-embed-v3-*)  : requires input as list + input_type="text"
      - OpenAI  (text-embedding-3-*) : accepts list input, no input_type needed

    Returns {"embeddings": [[float, …], …]} on success or {"error": str} on failure.
    """
    entry = GITHUB_MODEL_MAP.get(_canonical_gh_name(model_name), {})
    sdk_name = entry.get("sdk_name", model_name)
    token_kind = entry.get("token", "standard")

    if token_kind == "copilot":
        token = _get_copilot_token()
        endpoint = COPILOT_CHAT_ENDPOINT
        extra_headers = COPILOT_CHAT_HEADERS
    else:
        token = GITHUB_TOKEN
        endpoint = GITHUB_INFERENCE_ENDPOINT
        extra_headers = {}

    if not token:
        return {"error": "No token available for this model."}

    # Build request body — Cohere needs input_type, OpenAI-style doesn't mind it
    body: dict[str, object] = {"model": sdk_name, "input": inputs}
    if sdk_name in _COHERE_EMBED_FAMILY:
        body["input_type"] = "text"

    try:
        r = requests.post(
            f"{endpoint}/embeddings",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                **extra_headers,
            },
            json=body,
            timeout=30,
            verify=SSL_VERIFY,
        )
        r.raise_for_status()
        data = r.json()
        # OpenAI-shape: {"data": [{"embedding": [...], "index": 0}, ...]}
        vectors = [
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x.get("index", 0))
        ]
        return {"embeddings": vectors}
    except Exception as e:
        log.error("GH embed error for %s: %s", sdk_name, e)
        return {"error": str(e)}


def _handle_embed_request(model_name: str, data: dict[str, object]):
    """Shared logic for /api/embeddings and /api/embed.

    Priority:
      1. GH | / GC | embed models  → _gh_embed (free GitHub quota)
      2. BR | embed models          → _bedrock_embed (pay-per-use fallback)
      3. 155 | / bare names         → fall through to remote Ollama proxy
    """
    # Normalise input: accept both "input" (OpenAI style) and "prompt" (Ollama style)
    raw_input = data.get("input") or data.get("prompt", "")
    inputs: list[str] = (
        [str(t) for t in raw_input] if isinstance(raw_input, list) else [str(raw_input)]
    )

    if _is_gh_embed_model(model_name):
        canonical = _canonical_gh_name(model_name)
        result = _gh_embed(canonical, inputs)
        if "error" in result:
            return jsonify(result), 502
        # Return Ollama-compatible shape: single "embedding" for one input,
        # "embeddings" list for multiple.
        embeddings_list = cast("list[list[float]]", result["embeddings"])
        if len(inputs) == 1:
            return jsonify(
                {
                    "model": model_name,
                    "embedding": embeddings_list[0],
                    "embeddings": embeddings_list,
                }
            )
        return jsonify(
            {
                "model": model_name,
                "embeddings": embeddings_list,
            }
        )

    if is_bedrock_embed_model(model_name):
        canonical = _canonical_gh_name(model_name)
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            return jsonify({"error": "AWS credentials not configured."}), 500
        results = []
        for text in inputs:
            r = _bedrock_embed(canonical, text)
            if "error" in r:
                return jsonify(r), 502
            results.append(r["embedding"])
        if len(inputs) == 1:
            return jsonify(
                {
                    "model": model_name,
                    "embedding": results[0],
                    "embeddings": results,
                }
            )
        return jsonify({"model": model_name, "embeddings": results})

    return None  # Caller should fall through to Ollama proxy


@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    data = request.json or {}
    model_name = data.get("model", "")
    log.debug("/api/embeddings model=%r", model_name)
    br_resp = _handle_embed_request(model_name, data)
    if br_resp is not None:
        return br_resp
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
    resp = proxy_request("POST", "/api/embeddings", data=data)
    return make_proxy_response(resp)


@app.route("/api/embed", methods=["POST"])
def embed():
    data = request.json or {}
    model_name = data.get("model", "")
    log.debug("/api/embed model=%r", model_name)
    br_resp = _handle_embed_request(model_name, data)
    if br_resp is not None:
        return br_resp
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
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
                verify=SSL_VERIFY,
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
        verify=SSL_VERIFY,
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

    # Open the GitHub device page (pre-filled with the user code) and the
    # local auth status page in the default browser.
    device_url = f"{d['verification_uri']}?user_code={d['user_code']}"
    status_url = f"http://localhost:{PORT}/auth/status"
    threading.Timer(0.5, webbrowser.open, args=[device_url]).start()
    threading.Timer(1.0, webbrowser.open, args=[status_url]).start()

    return jsonify(
        {
            "status": "pending",
            "message": (
                f"Opening browser automatically.\n"
                f"If it doesn't open, go to {d['verification_uri']} "
                f"and enter code: {d['user_code']}"
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

    log.info("Device flow polling started")
    expires_at = float(cast(float, flow["expires_at"]))
    poll_interval = float(cast(float, flow["interval"]))
    while time.time() < expires_at:
        time.sleep(poll_interval)
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
                verify=SSL_VERIFY,
            )
            data = r.json()
        except Exception as e:
            log.warning("Device flow poll error: %s", e)
            continue

        if "access_token" in data:
            token = str(data["access_token"])
            _set_copilot_token(token)
            _pending_device_flow = {}
            log.info("Copilot OAuth token obtained — rebuilding model catalogue")
            # Reuse the already-discovered GH | models so we don't re-probe them.
            existing_gh = [
                m for m in GITHUB_MODELS if str(m["name"]).startswith(GH_PREFIX)
            ]
            _rebuild_catalogue(existing_gh_models=existing_gh if existing_gh else None)
            return
        elif str(data.get("error", "")) == "slow_down":
            poll_interval += 5
        elif str(data.get("error", "")) == "authorization_pending":
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

    gh_chat = sum(
        1
        for m in GITHUB_MODELS
        if m.get("token") == "standard" and m.get("kind") == "chat"
    )
    gh_embed = sum(
        1
        for m in GITHUB_MODELS
        if m.get("token") == "standard" and m.get("kind") == "embed"
    )
    gc_chat = sum(
        1
        for m in GITHUB_MODELS
        if m.get("token") == "copilot" and m.get("kind") == "chat"
    )
    gc_embed = sum(
        1
        for m in GITHUB_MODELS
        if m.get("token") == "copilot" and m.get("kind") == "embed"
    )
    br_chat_count = sum(
        1
        for m in GITHUB_MODELS
        if m.get("token") == "bedrock" and m.get("kind") == "chat"
    )
    br_embed_count = sum(
        1
        for m in GITHUB_MODELS
        if m.get("token") == "bedrock" and m.get("kind") == "embed"
    )
    br_count = br_chat_count + br_embed_count
    if not _BOTO3_AVAILABLE:
        br_status = "unavailable — install boto3 (pip install boto3)"
    elif not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        br_status = (
            "not configured — add AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY to .env"
        )
    else:
        br_status = f"configured — {br_count} models ({br_chat_count} chat, {br_embed_count} embed) in {AWS_BEDROCK_REGION}"

    status = "healthy" if remote_status == "ok" else "degraded"

    return jsonify(
        {
            "status": status,
            "remote_ollama": remote_status,
            "github_models": f"{gh_status} ({gh_chat} chat, {gh_embed} embed)",
            "copilot_models": f"{gc_status} ({gc_chat} chat, {gc_embed} embed)",
            "bedrock_models": br_status,
        }
    )


@app.route("/", methods=["GET"])
def root():
    br_count = sum(1 for m in GITHUB_MODELS if m.get("token") == "bedrock")
    gh_count = sum(
        1 for m in GITHUB_MODELS if m.get("token") not in ("copilot", "bedrock")
    )
    gc_count = sum(1 for m in GITHUB_MODELS if m.get("token") == "copilot")
    return jsonify(
        {
            "message": "Ollama Proxy with GitHub Models + AWS Bedrock",
            "version": "1.6.0",
            "model_counts": {
                "gh": gh_count,
                "gc": gc_count,
                "br": br_count,
            },
            "endpoints": [
                "GET    /api/tags              — list all models",
                "POST   /api/show              — model metadata",
                "POST   /api/chat              — Ollama-style chat (NDJSON)",
                "POST   /api/generate          — Ollama-style generate (NDJSON)",
                "POST   /api/chat/completions  — OpenAI-compat chat completions",
                "POST   /api/embeddings        — embeddings (Bedrock Titan or remote Ollama)",
                "POST   /api/embed             — embeddings (Bedrock Titan or remote Ollama)",
                "POST   /api/pull              — pull a model",
                "GET    /api/ps                — running models",
                "DELETE /api/delete            — delete a model",
                "POST   /api/copy              — copy a model",
                "GET    /v1/models             — OpenAI-compat model list",
                "POST   /v1/chat/completions   — OpenAI-compat chat completions",
                "POST   /v1/embeddings         — OpenAI-compat embeddings (GH/GC/BR/Ollama)",
                "GET    /auth/copilot          — start Copilot browser login (device flow)",
                "GET    /auth/status           — check Copilot login state",
                "GET    /health                — health check",
            ],
        }
    )


# ---------------------------------------------------------------------------
# /v1/models  and  /v1/chat/completions  (OpenAI-compat surface for Continue.dev)
# ---------------------------------------------------------------------------


def _all_models_as_openai() -> list[dict[str, object]]:
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
        token_kind = m.get("token", "github")
        kind = m.get("kind", "chat")
        owned_by = (
            "amazon-bedrock"
            if token_kind == "bedrock"
            else "github-copilot"
            if token_kind == "copilot"
            else "github"
        )
        models.append(
            {
                "id": m["name"],
                "object": "model",
                "created": 0,
                "owned_by": owned_by,
                # Non-standard but useful for clients that inspect capabilities
                "capabilities": {"type": kind},
            }
        )

    return models


@app.route("/v1/models", methods=["GET"])
def v1_models():
    """OpenAI-compat GET /v1/models — lists all proxied models."""
    return jsonify({"object": "list", "data": _all_models_as_openai()})


@app.route("/v1/embeddings", methods=["POST"])
def v1_embeddings():
    """OpenAI-compat POST /v1/embeddings — routes to GH, Bedrock, or remote Ollama."""
    data = request.json or {}
    model_name = data.get("model", "")
    log.debug("/v1/embeddings model=%r", model_name)
    resp = _handle_embed_request(model_name, data)
    if resp is not None:
        return resp
    # Fall through to remote Ollama's OpenAI-compat embeddings endpoint
    if model_name.startswith(REMOTE_PREFIX):
        data["model"] = model_name[len(REMOTE_PREFIX) :]
    proxy_resp = proxy_request("POST", "/v1/embeddings", data=data)
    return make_proxy_response(proxy_resp)


def _github_chat_streaming_openai(model_name: str, messages: list[dict[str, object]]):
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

    if is_any_embed_model(model_name):
        return jsonify(
            {
                "error": {
                    "message": (
                        f"'{model_name}' is an embedding model. "
                        "Use POST /v1/embeddings instead."
                    ),
                    "type": "invalid_request_error",
                    "code": "model_not_supported_for_chat",
                }
            }
        ), 400

    if is_bedrock_model(model_name):
        model_name = _canonical_gh_name(model_name)
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            return jsonify(
                {
                    "error": {
                        "message": "AWS credentials not configured.",
                        "type": "api_error",
                    }
                }
            ), 500

        if do_stream:
            return Response(
                stream_with_context(
                    _bedrock_chat_streaming_openai(model_name, messages)
                ),
                status=200,
                content_type="text/event-stream",
            )

        result = _bedrock_chat_blocking(model_name, messages)
        if "error" in result:
            return jsonify(
                {"error": {"message": result["error"], "type": "api_error"}}
            ), 502

        return jsonify(
            {
                "id": "chatcmpl-bedrock",
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
                    "total_tokens": cast(int, result.get("prompt_eval_count", 0))
                    + cast(int, result.get("eval_count", 0)),
                },
            }
        )

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
                    "total_tokens": cast(int, result.get("prompt_eval_count", 0))
                    + cast(int, result.get("eval_count", 0)),
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
