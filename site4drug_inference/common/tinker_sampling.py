#!/usr/bin/env python3
"""Helpers for building Tinker sampling params with optional seed support."""

from __future__ import annotations

import inspect
from typing import Any


def sampling_seed_param_name(types_module: Any) -> str | None:
    sampling_params_cls = getattr(types_module, "SamplingParams", None)
    if sampling_params_cls is None:
        return None
    try:
        signature = inspect.signature(sampling_params_cls)
    except (TypeError, ValueError):
        return None
    parameters = signature.parameters
    for name in ("seed", "random_seed"):
        if name in parameters:
            return name
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return "seed"
    return None


def sampling_seed_supported(types_module: Any) -> bool:
    return sampling_seed_param_name(types_module) is not None


def build_sampling_params(
    types_module: Any,
    *,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None = None,
    sampling_seed: int | None = None,
) -> tuple[Any, bool]:
    kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stop": list(stop or []),
    }
    seed_param = sampling_seed_param_name(types_module)
    supported = seed_param is not None
    if sampling_seed is not None and supported:
        kwargs[seed_param] = int(sampling_seed)
    return types_module.SamplingParams(**kwargs), supported
