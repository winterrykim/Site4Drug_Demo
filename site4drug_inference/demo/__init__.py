"""Demo and CLI helpers for Site4Drug inference."""

from __future__ import annotations

__all__ = ["run_prediction"]


def __getattr__(name: str):
    """Lazily expose the main prediction helper without eager CLI imports."""
    if name == "run_prediction":
        from .predict_site import run_prediction

        return run_prediction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
