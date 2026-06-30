"""Shared helpers for constructing Engram SDK clients.

The Engram SDK requires an API key and does not read it from the environment
itself, so these helpers add the conventional ``ENGRAM_API_KEY`` fallback that
LangChain integrations are expected to support.
"""

from __future__ import annotations

import os
from typing import Any

from engram import AsyncEngramClient, EngramClient

_API_KEY_ENV = "ENGRAM_API_KEY"
_BASE_URL_ENV = "ENGRAM_BASE_URL"


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve the Engram API key from an argument or the environment.

    Args:
        api_key: Explicit API key. If `None`, the `ENGRAM_API_KEY` environment
            variable is used.

    Returns:
        The resolved API key.

    Raises:
        ValueError: If no API key can be resolved.
    """
    resolved = api_key or os.environ.get(_API_KEY_ENV)
    if not resolved:
        msg = (
            "An Engram API key is required. Pass `api_key=...` or set the "
            f"`{_API_KEY_ENV}` environment variable. Create a key at "
            "https://console.weaviate.cloud/engram."
        )
        raise ValueError(msg)
    return resolved


def build_client(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    timeout: float | None = None,
) -> EngramClient:
    """Build a synchronous `EngramClient`.

    Args:
        api_key: Engram API key. Falls back to `ENGRAM_API_KEY`.
        base_url: Override the API base URL. Falls back to `ENGRAM_BASE_URL`,
            then the SDK default.
        timeout: Per-request timeout in seconds. Uses the SDK default if `None`.

    Returns:
        A configured `EngramClient`.
    """
    kwargs = _client_kwargs(api_key, base_url=base_url, timeout=timeout)
    return EngramClient(**kwargs)


def build_async_client(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    timeout: float | None = None,
) -> AsyncEngramClient:
    """Build an asynchronous `AsyncEngramClient`.

    Args:
        api_key: Engram API key. Falls back to `ENGRAM_API_KEY`.
        base_url: Override the API base URL. Falls back to `ENGRAM_BASE_URL`,
            then the SDK default.
        timeout: Per-request timeout in seconds. Uses the SDK default if `None`.

    Returns:
        A configured `AsyncEngramClient`.
    """
    kwargs = _client_kwargs(api_key, base_url=base_url, timeout=timeout)
    return AsyncEngramClient(**kwargs)


def _client_kwargs(
    api_key: str | None,
    *,
    base_url: str | None,
    timeout: float | None,
) -> dict[str, Any]:
    """Assemble keyword arguments shared by both client constructors."""
    kwargs: dict[str, Any] = {"api_key": _resolve_api_key(api_key)}
    resolved_base_url = base_url or os.environ.get(_BASE_URL_ENV)
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs
