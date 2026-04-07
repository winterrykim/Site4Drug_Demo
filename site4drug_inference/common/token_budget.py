#!/usr/bin/env python3
"""Token-budget helpers for overflow-safe prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class BudgetDecision:
    """Result of prompt budget evaluation."""

    strategy: str
    input_tokens: int
    max_input_tokens: int
    overflow_tokens: int
    reason: str

    @property
    def is_within_budget(self) -> bool:
        return self.overflow_tokens <= 0


def prompt_input_length(renderer, messages: Sequence[dict]) -> int:
    """Build generation prompt and return token length."""
    model_input = renderer.build_generation_prompt(list(messages))
    return int(getattr(model_input, "length", 0))


def evaluate_budget(
    strategy: str,
    renderer,
    messages: Sequence[dict],
    max_input_tokens: int,
) -> BudgetDecision:
    """Evaluate whether a prompt fits the configured input token budget."""
    token_count = prompt_input_length(renderer=renderer, messages=messages)
    overflow = token_count - max_input_tokens
    if overflow <= 0:
        reason = "within_budget"
    else:
        reason = f"exceeds_budget_by_{overflow}"
    return BudgetDecision(
        strategy=strategy,
        input_tokens=token_count,
        max_input_tokens=max_input_tokens,
        overflow_tokens=max(overflow, 0),
        reason=reason,
    )
