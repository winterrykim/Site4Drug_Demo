#!/usr/bin/env python3
"""Lightweight ReAct-style orchestration trace helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class OrchestratorStep:
    step_index: int
    plan: str
    execution: str
    observation: str
    status: str
    retry_index: int = 0
    error_code: str | None = None
    timestamp_utc: str = ""


class LightweightReActOrchestrator:
    """Deterministic ReAct trace collector with bounded retries."""

    def __init__(
        self,
        max_steps: int = 8,
        max_retries: int = 2,
        step_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.max_steps = max(1, int(max_steps))
        self.max_retries = max(0, int(max_retries))
        self._steps: list[OrchestratorStep] = []
        self._retry_counts: dict[str, int] = {}
        self._step_callback = step_callback

    def can_continue(self) -> bool:
        return len(self._steps) < self.max_steps

    def can_retry(self, retry_key: str) -> bool:
        used = self._retry_counts.get(retry_key, 0)
        return used < self.max_retries

    def note_retry(self, retry_key: str) -> int:
        used = self._retry_counts.get(retry_key, 0) + 1
        self._retry_counts[retry_key] = used
        return used

    def record(
        self,
        *,
        plan: str,
        execution: str,
        observation: str,
        status: str,
        retry_index: int = 0,
        error_code: str | None = None,
    ) -> None:
        if not self.can_continue():
            return
        self._steps.append(
            OrchestratorStep(
                step_index=len(self._steps) + 1,
                plan=plan,
                execution=execution,
                observation=observation,
                status=status,
                retry_index=retry_index,
                error_code=error_code,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        if self._step_callback is not None:
            step = self._steps[-1]
            try:
                self._step_callback(
                    {
                        "event_type": "orchestrator_step",
                        "step_key": f"orchestrator_step_{step.step_index}",
                        "label": step.plan,
                        "status": str(step.status),
                        "timestamp_utc": step.timestamp_utc,
                        "details": {
                            "step_index": step.step_index,
                            "execution": step.execution,
                            "observation": step.observation,
                            "retry_index": step.retry_index,
                            "error_code": step.error_code,
                        },
                    }
                )
            except Exception:
                # Progress callbacks are best-effort and must not break inference.
                pass

    def to_list(self) -> list[dict]:
        return [asdict(step) for step in self._steps]
