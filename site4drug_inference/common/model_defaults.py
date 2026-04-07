#!/usr/bin/env python3
"""Runtime model/checkpoint defaults with environment-variable overrides."""

from __future__ import annotations

import os

DEFAULT_BASE_MODEL_FALLBACK = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_CHECKPOINT_FALLBACK = (
    "tinker://9b162e30-efa9-5518-ac42-8b8979b3a3e6:train:0/"
    "sampler_weights/best_epoch1_step400"
)

# Archived previous production defaults (kept for rollback/reference):
# - Checkpoint: tinker://9b162e30-efa9-5518-ac42-8b8979b3a3e6:train:0/sampler_weights/epoch3_step2500
# - Base model: Qwen/Qwen3-30B-A3B
# - Checkpoint: tinker://69f75125-5df9-5ce0-b973-b1a50890524a:train:0/sampler_weights/best_epoch8_step8400


def get_base_model() -> str:
    """Return the default base model, optionally overridden by env."""
    return os.environ.get("SITE4DRUG_BASE_MODEL", DEFAULT_BASE_MODEL_FALLBACK).strip()


def get_default_checkpoint() -> str:
    """Return the default checkpoint, optionally overridden by env."""
    return os.environ.get("SITE4DRUG_CHECKPOINT", DEFAULT_CHECKPOINT_FALLBACK).strip()


BASE_MODEL = get_base_model()
DEFAULT_CHECKPOINT = get_default_checkpoint()
