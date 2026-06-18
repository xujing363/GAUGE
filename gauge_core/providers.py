"""Multi-provider LLM configuration for the GAUGE Assistant.

The Assistant talks to any OpenAI-API-compatible chat-completions endpoint
(this is how the original DeepSeek-only agent already worked). This module
generalises that single hard-coded client into a small registry of named
providers, so the user can pick a faster chat model, a stronger reasoning
model, or OpenAI, and so the agent can transparently *fall back* to another
configured provider if the primary one errors (e.g. a bad key or an outage).

No provider here is special-cased in the calling code: each is just a
``base_url`` + an environment variable holding the key + a default model
name. Adding a new OpenAI-compatible vendor is one entry in ``PROVIDERS``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ._env import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMProvider:
    """One OpenAI-API-compatible chat endpoint."""

    name: str
    label: str
    base_url: str
    env_key: str
    default_model: str
    supports_tools: bool = True
    # Reasoning models (e.g. deepseek-reasoner) reject ``temperature`` and are
    # slower but stronger -- the deep-report mode prefers one of these.
    is_reasoning: bool = False


PROVIDERS: dict[str, LLMProvider] = {
    "deepseek-chat": LLMProvider(
        name="deepseek-chat",
        label="DeepSeek Chat (fast, default)",
        base_url="https://api.deepseek.com",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    "deepseek-reasoner": LLMProvider(
        name="deepseek-reasoner",
        label="DeepSeek Reasoner (deep reasoning)",
        base_url="https://api.deepseek.com",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-reasoner",
        # Reasoning models do NOT support function/tool calling, so they can only
        # be used for the report's tool-less synthesis step -- never as the chat
        # (tool-using) provider. Hidden from the chat-model selector for that reason.
        supports_tools=False,
        is_reasoning=True,
    ),
    "openai": LLMProvider(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o",
    ),
}

DEFAULT_PROVIDER = os.environ.get("GAUGE_ASSISTANT_PROVIDER", "deepseek-chat")
# The report mode prefers a stronger reasoning model when one is configured.
DEFAULT_REPORT_PROVIDER = os.environ.get("GAUGE_REPORT_PROVIDER", "deepseek-reasoner")


def get_provider(name: str) -> LLMProvider:
    if name not in PROVIDERS:
        raise KeyError(f"Unknown LLM provider {name!r}. Known: {sorted(PROVIDERS)}")
    return PROVIDERS[name]


def provider_base_url(provider: LLMProvider) -> str:
    """Allow a per-provider base-url override via ``<ENV_KEY without _API_KEY>_BASE_URL``."""
    override_key = provider.env_key.replace("_API_KEY", "_BASE_URL")
    return os.environ.get(override_key, provider.base_url)


def resolve_key(provider: LLMProvider, api_key: str | None = None) -> str | None:
    """Explicit key wins, else the provider's env var (incl. values loaded from .env)."""
    return api_key or os.environ.get(provider.env_key) or None


def build_client(provider: LLMProvider, api_key: str | None = None) -> tuple[Any, str]:
    """Return ``(OpenAI client, model_name)`` for a provider.

    Raises ``LookupError`` if no key is available so callers can present a
    friendly "configure a key" message instead of an opaque auth failure.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'openai' package is required for the GAUGE Assistant.") from exc

    key = resolve_key(provider, api_key)
    if not key:
        raise LookupError(
            f"No API key for provider {provider.name!r}. Set {provider.env_key} "
            "in your environment / .env, or paste a key in the sidebar."
        )
    # Reasoning models (deep report) can take minutes; give them a generous but
    # finite timeout so a hung request surfaces an error instead of blocking the UI.
    timeout = 300.0 if provider.is_reasoning else 120.0
    client = OpenAI(api_key=key, base_url=provider_base_url(provider), timeout=timeout, max_retries=1)
    return client, provider.default_model


def chat_completion(
    primary: LLMProvider,
    *,
    fallbacks: list[LLMProvider] | None = None,
    api_keys: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``chat.completions.create`` on ``primary``, falling back in order.

    ``api_keys`` maps ``provider.name -> key`` (e.g. keys pasted in the UI).
    Reasoning models do not accept ``temperature``; it is dropped automatically
    for those. The first provider that both has a key and answers without an
    exception wins; the last exception is re-raised if all fail.
    """
    api_keys = api_keys or {}
    chain = [primary, *(fallbacks or [])]
    last_exc: Exception | None = None
    for provider in chain:
        try:
            client, model = build_client(provider, api_keys.get(provider.name))
        except LookupError as exc:  # no key for this provider -- skip to next
            last_exc = exc
            continue
        call_kwargs = dict(kwargs)
        if provider.is_reasoning:
            call_kwargs.pop("temperature", None)
        try:
            return client.chat.completions.create(model=model, **call_kwargs)
        except Exception as exc:  # noqa: BLE001 - try the next provider on any API error
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No LLM provider was available to handle the request.")
