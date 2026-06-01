#!/usr/bin/env python3
"""OpenRouter chat-completions client for OpenAI-compatible inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request
from typing import Any, Sequence


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_APP_TITLE = "Site4Drug Demo"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"


@dataclass(frozen=True)
class ApproxPrompt:
    """Prompt-like object exposing a length field for token-budget checks."""

    text: str
    length: int


class ApproxChatRenderer:
    """Minimal renderer used when the backend accepts structured chat messages."""

    def build_generation_prompt(self, messages: Sequence[dict[str, Any]]) -> ApproxPrompt:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip() or "user"
            content = str(message.get("content", ""))
            lines.append(f"{role}: {content}")
        text = "\n\n".join(lines)
        # Conservative approximation for OpenRouter budget checks without a
        # provider-specific tokenizer.
        approx_tokens = max(1, len(text) // 4)
        return ApproxPrompt(text=text, length=approx_tokens)

    def get_stop_sequences(self) -> list[str]:
        return []


class OpenRouterChatClient:
    """Small raw-HTTP client for OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        referer: str | None = None,
        title: str | None = DEFAULT_OPENROUTER_APP_TITLE,
        timeout: float = 120.0,
    ) -> None:
        api_key = str(api_key or "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter inference.")
        model = str(model or "").strip()
        if not model:
            raise ValueError("OPENROUTER_MODEL or --openrouter-model is required for OpenRouter inference.")
        self.api_key = api_key
        self.model = model
        self.base_url = str(base_url or DEFAULT_OPENROUTER_BASE_URL).strip().rstrip("/")
        self.referer = str(referer or "").strip()
        self.title = str(title or "").strip()
        self.timeout = float(timeout)

    @property
    def endpoint_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def sample_messages(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stop: Sequence[str] | None = None,
        sampling_seed: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
                for message in messages
            ],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        payload["model"] = self.model
        if stop:
            payload["stop"] = list(stop)
        if sampling_seed is not None:
            payload["seed"] = int(sampling_seed)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-OpenRouter-Title"] = self.title

        request = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed ({exc.code}): {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenRouter returned non-JSON response.") from exc

        choices = decoded.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter response contained no choices: {response_body[:500]}")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    return "".join(parts)
        text = first.get("text")
        if isinstance(text, str):
            return text
        raise RuntimeError(f"OpenRouter response did not include text content: {response_body[:500]}")
