#!/usr/bin/env python3
"""Runtime helpers for local environment bootstrap."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(env_path: Path) -> dict[str, str]:
    """Parse a simple shell-style env file into key/value pairs."""
    parsed: dict[str, str] = {}
    if not env_path.exists():
        return parsed

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_outer_quotes(value.strip())
        if key:
            parsed[key] = value

    return parsed


def ensure_tinker_api_key(repo_root: Path, env_filename: str = ".tinker.env") -> bool:
    """Ensure TINKER_API_KEY is present, auto-loading from a local env file if needed."""
    if os.environ.get("TINKER_API_KEY"):
        return True

    env_path = repo_root / env_filename
    env_values = load_env_file(env_path)
    api_key = env_values.get("TINKER_API_KEY")
    if not api_key:
        return False

    os.environ["TINKER_API_KEY"] = api_key
    return True


def ensure_openrouter_api_key(repo_root: Path, env_filename: str = ".openrouter.env") -> bool:
    """Ensure OPENROUTER_API_KEY is present, auto-loading from a local env file if needed."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return True

    env_path = repo_root / env_filename
    env_values = load_env_file(env_path)
    api_key = env_values.get("OPENROUTER_API_KEY")
    if not api_key:
        return False

    os.environ["OPENROUTER_API_KEY"] = api_key
    for optional_key in (
        "OPENROUTER_MODEL",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_TITLE",
    ):
        if optional_key in env_values and not os.environ.get(optional_key):
            os.environ[optional_key] = env_values[optional_key]
    return True
