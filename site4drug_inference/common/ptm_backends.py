#!/usr/bin/env python3
"""PTM extraction backend abstraction for Site4Drug."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from site4drug_inference.common.musitedeep_api import DEFAULT_MUSITEDEEP_API_BASE_URL, run_musitedeep_api_predictions

DEFAULT_MUSITEDEEP_MODEL_MAP = Path(__file__).resolve().with_name("musitedeep_model_map.json")
PTM_TYPE_N_LINKED = "N-linked_glycosylation"


class PTMBackendError(RuntimeError):
    """Raised when the selected PTM backend cannot be used."""


def _summarize_ptm_sites(ptm_sites: list[Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    masked_positions: set[int] = set()
    for site in ptm_sites:
        ptm_type = str(getattr(site, "ptm_type", "Unknown"))
        counts[ptm_type] = counts.get(ptm_type, 0) + 1
        start = int(getattr(site, "mask_start", 0))
        end = int(getattr(site, "mask_end", 0))
        if start > 0 and end >= start:
            masked_positions.update(range(start, end + 1))
    return {
        "total_sites": len(ptm_sites),
        "counts_by_type": counts,
        "n_masked_positions": len(masked_positions),
    }


def _dedupe_sites(sites: list[Any]) -> list[Any]:
    dedup: dict[tuple[str, int], Any] = {}
    for site in sites:
        key = (str(getattr(site, "ptm_type", "")), int(getattr(site, "position", -1)))
        if key not in dedup:
            dedup[key] = site
    out = list(dedup.values())
    out.sort(key=lambda s: (int(getattr(s, "position", 0)), str(getattr(s, "ptm_type", ""))))
    return out


def _load_model_specs(path: str | Path | None = None) -> dict[str, Any]:
    model_map = Path(path) if path else DEFAULT_MUSITEDEEP_MODEL_MAP
    payload = json.loads(model_map.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("musitedeep model map must be a JSON object")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("musitedeep model map field 'models' must be a list")
    return {
        "path": model_map,
        "version": str(payload.get("version", "unknown")),
        "models": [m for m in models if isinstance(m, dict)],
    }


def _normalized_source(source: str) -> str:
    src = str(source or "musitedeep").strip().lower()
    if src in {"musitedeep", "hybrid", "multi_rule", "glyco_only"}:
        return src
    return "musitedeep"


def extract_ptm_sites(
    *,
    seq: str,
    pad: int,
    source: str,
    site_builder: Callable[..., Any],
    multi_rule_finder: Callable[[str, int, str], list[Any]],
    glyco_only_finder: Callable[[str, int, str], list[Any]],
    musitedeep_api_base_url: str = DEFAULT_MUSITEDEEP_API_BASE_URL,
    musitedeep_model_map: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Resolve PTM sites from selected backend with provenance."""
    selected = _normalized_source(source)
    warnings: list[str] = []
    backend_meta: dict[str, Any] = {
        "ptm_backend_selected": selected,
        "ptm_backend_effective": selected,
        "musitedeep_available": False,
        "musitedeep_status": "not_requested",
        "musitedeep_api_base_url": str(musitedeep_api_base_url or DEFAULT_MUSITEDEEP_API_BASE_URL).strip().rstrip("/"),
        "musitedeep_endpoint_urls": [],
        "musitedeep_models_attempted": [],
        "musitedeep_models_succeeded": [],
        "musitedeep_error_summary": "",
        "musitedeep_raw_calls": [],
        "ptm_rule_version": "rulepack_v1",
    }

    if selected == "multi_rule":
        sites = multi_rule_finder(seq, pad, "multi_rule")
        return {
            "ptm_sites": _dedupe_sites(sites),
            "ptm_summary": _summarize_ptm_sites(sites),
            "backend": backend_meta,
            "warnings": warnings,
        }

    if selected == "glyco_only":
        sites = glyco_only_finder(seq, pad, "glyco_only")
        return {
            "ptm_sites": _dedupe_sites(sites),
            "ptm_summary": _summarize_ptm_sites(sites),
            "backend": backend_meta,
            "warnings": warnings,
        }

    model_cfg = _load_model_specs(musitedeep_model_map)
    backend_meta["ptm_rule_version"] = f"musitedeep:{model_cfg['version']}"
    musite = run_musitedeep_api_predictions(
        sequence=seq,
        model_specs=model_cfg["models"],
        base_url=str(musitedeep_api_base_url or DEFAULT_MUSITEDEEP_API_BASE_URL).strip().rstrip("/"),
    )
    backend_meta["musitedeep_available"] = bool(musite.get("available"))
    backend_meta["musitedeep_status"] = str(musite.get("status", "unknown"))
    backend_meta["musitedeep_models_attempted"] = list(musite.get("models_attempted", []))
    backend_meta["musitedeep_models_succeeded"] = list(musite.get("models_succeeded", []))
    backend_meta["musitedeep_api_base_url"] = str(musite.get("api_base_url", ""))
    backend_meta["musitedeep_endpoint_urls"] = list(musite.get("endpoint_urls", []))
    backend_meta["musitedeep_raw_calls"] = list(musite.get("raw_calls", []) or [])
    backend_meta["musitedeep_chunking"] = dict(musite.get("chunking", {}) or {})
    errors = musite.get("errors", {}) or {}
    backend_meta["musitedeep_error_summary"] = ";".join(f"{k}:{v}" for k, v in errors.items())[:1000]

    glyco_sites = glyco_only_finder(seq, pad, "glyco_only")
    predicted_sites: list[Any] = []
    pred_by_type = musite.get("predictions_by_type", {}) or {}
    for ptm_type, positions in pred_by_type.items():
        for pos in positions:
            predicted_sites.append(
                site_builder(
                    ptm_type=ptm_type,
                    position=int(pos),
                    mask_start=max(1, int(pos) - pad),
                    mask_end=min(len(seq), int(pos) + pad),
                    rule_confidence="medium",
                    source="musitedeep_api",
                )
            )

    musitedeep_ok = bool(musite.get("available")) and str(musite.get("status")) in {"ok", "ok_empty"}
    if selected == "musitedeep":
        sites = _dedupe_sites([*predicted_sites, *glyco_sites])
        backend_meta["ptm_backend_effective"] = "musitedeep_plus_glyco"
        if not musitedeep_ok:
            msg = (
                f"musitedeep backend unavailable/failed (status={backend_meta['musitedeep_status']}, "
                f"errors={backend_meta['musitedeep_error_summary'] or 'none'})"
            )
            warnings.append(msg)
            if strict:
                raise PTMBackendError(msg)
        return {
            "ptm_sites": sites,
            "ptm_summary": _summarize_ptm_sites(sites),
            "backend": backend_meta,
            "warnings": warnings,
        }

    # hybrid
    if musitedeep_ok:
        sites = _dedupe_sites([*predicted_sites, *glyco_sites])
        backend_meta["ptm_backend_effective"] = "hybrid_musitedeep_plus_glyco"
        return {
            "ptm_sites": sites,
            "ptm_summary": _summarize_ptm_sites(sites),
            "backend": backend_meta,
            "warnings": warnings,
        }

    warnings.append(
        "musitedeep_unavailable_for_hybrid_fallback;using_multi_rule"
    )
    sites = _dedupe_sites(multi_rule_finder(seq, pad, "multi_rule"))
    backend_meta["ptm_backend_effective"] = "hybrid_fallback_multi_rule"
    return {
        "ptm_sites": sites,
        "ptm_summary": _summarize_ptm_sites(sites),
        "backend": backend_meta,
        "warnings": warnings,
    }
