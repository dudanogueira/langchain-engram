"""Unit tests for client construction helpers."""

from __future__ import annotations

import pytest

from langchain_engram._client import build_async_client, build_client


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENGRAM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Engram API key is required"):
        build_client()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGRAM_API_KEY", "sk-test")
    client = build_client()
    assert client is not None


def test_explicit_api_key_and_base_url() -> None:
    client = build_client("sk-test", base_url="https://example.test")
    assert client is not None


async def test_async_client_builds() -> None:
    client = build_async_client("sk-test")
    assert client is not None
