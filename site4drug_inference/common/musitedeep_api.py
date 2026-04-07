#!/usr/bin/env python3
"""MusiteDeep HTTPS API client helpers."""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any

import requests

DEFAULT_MUSITEDEEP_API_BASE_URL = (
    os.environ.get("SITE4DRUG_MUSITEDEEP_API_BASE_URL", "https://www.musite.net").strip().rstrip("/")
)
# MusiteDeep docs mention a 1000-aa limit, but in practice requests at length
# 1000 may be rejected by the service. Use a conservative max to avoid hard
# failures in strict publication mode.
MUSITEDEEP_MAX_SEQUENCE_LENGTH = 999


def _normalize_label(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "_")


def _extract_cutoff_key(entry: dict[str, Any]) -> str | None:
    for key in entry.keys():
        if str(key).strip().lower().startswith("cutoff"):
            return str(key)
    return None


def _parse_scored_items(text: str) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for token in str(text or "").split(";"):
        token = token.strip()
        if not token or ":" not in token:
            continue
        label, value = token.split(":", 1)
        try:
            score = float(value.strip())
        except ValueError:
            continue
        items.append((label.strip(), score))
    return items


def _extract_positive_positions(
    *,
    results: list[dict[str, Any]],
    score_aliases: list[str],
    score_cutoff: float = 0.5,
) -> list[int]:
    aliases = {_normalize_label(a) for a in score_aliases if str(a).strip()}
    positives: set[int] = set()
    for row in results:
        if not isinstance(row, dict):
            continue
        try:
            pos = int(row.get("Position"))
        except (TypeError, ValueError):
            continue

        cutoff_key = _extract_cutoff_key(row)
        cutoff_text = str(row.get(cutoff_key, "")).strip() if cutoff_key else ""
        if cutoff_text and cutoff_text.lower() != "none":
            scored = _parse_scored_items(cutoff_text)
            if any(_normalize_label(label) in aliases for label, _ in scored):
                positives.add(pos)
                continue

        all_scores = _parse_scored_items(str(row.get("PTMscores", "")))
        if any(_normalize_label(label) in aliases and score >= score_cutoff for label, score in all_scores):
            positives.add(pos)
    return sorted(positives)


def _fetch_combined_result(
    *,
    base_url: str,
    model_names: list[str],
    sequence: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    model_segment = ";".join(str(m).strip() for m in model_names if str(m).strip())
    encoded_model = urllib.parse.quote(model_segment, safe=";")
    encoded_seq = urllib.parse.quote(sequence, safe="")
    url = f"{base_url}/musitedeep/{encoded_model}/{encoded_seq}"
    try:
        resp = requests.get(url, timeout=max(int(timeout_seconds), 5))
        status_code = int(resp.status_code)
        text = resp.text
    except requests.RequestException as exc:
        return {
            "ok": False,
            "url": url,
            "error": f"request_failed:{exc}",
            "status_code": None,
            "response_text": "",
            "response_json": None,
        }

    if status_code != 200:
        return {
            "ok": False,
            "url": url,
            "error": f"http_status_{status_code}",
            "status_code": status_code,
            "response_text": text,
            "response_json": None,
        }

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "url": url,
            "error": "invalid_json_response",
            "status_code": status_code,
            "response_text": text,
            "response_json": None,
        }

    if isinstance(payload, dict) and payload.get("Error"):
        return {
            "ok": False,
            "url": url,
            "error": f"api_error:{payload.get('Error')}",
            "status_code": status_code,
            "response_text": text,
            "response_json": payload,
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "url": url,
            "error": "unexpected_payload_type",
            "status_code": status_code,
            "response_text": text,
            "response_json": payload,
        }
    results = payload.get("Results", [])
    if not isinstance(results, list):
        return {
            "ok": False,
            "url": url,
            "error": "missing_results_list",
            "status_code": status_code,
            "response_text": text,
            "response_json": payload,
        }
    return {
        "ok": True,
        "url": url,
        "results": results,
        "status_code": status_code,
        "response_text": text,
        "response_json": payload,
    }


def _raw_call_from_fetch(
    raw: dict[str, Any],
    *,
    model_names: list[str],
) -> dict[str, Any]:
    return {
        "model_names": [str(m) for m in model_names],
        "url": str(raw.get("url", "")),
        "ok": bool(raw.get("ok")),
        "status_code": raw.get("status_code"),
        "error": str(raw.get("error", "")),
        "response_text": str(raw.get("response_text", "")),
    }


def _run_single_sequence_predictions(
    *,
    sequence: str,
    model_specs: list[dict[str, Any]],
    base_url: str = DEFAULT_MUSITEDEEP_API_BASE_URL,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Run MusiteDeep PTM predictions for one sequence (<= API max length)."""
    attempted: list[str] = []
    succeeded: list[str] = []
    errors: dict[str, str] = {}
    predictions_by_type: dict[str, list[int]] = {}
    endpoint_urls: list[str] = []
    raw_calls: list[dict[str, Any]] = []

    enabled_specs: list[dict[str, Any]] = []
    api_model_order: list[str] = []
    api_model_seen: set[str] = set()
    for spec in model_specs:
        if not isinstance(spec, dict):
            continue
        ptm_type = str(spec.get("ptm_type", "")).strip()
        if not ptm_type:
            continue
        if spec.get("enabled", True) is False:
            continue
        attempted.append(ptm_type)
        api_model = str(spec.get("api_model", ptm_type)).strip() or ptm_type
        if api_model not in api_model_seen:
            api_model_seen.add(api_model)
            api_model_order.append(api_model)
        enabled_specs.append(dict(spec))

    if not api_model_order:
        return {
            "available": False,
            "status": "failed",
            "predictions_by_type": predictions_by_type,
            "models_attempted": attempted,
            "models_succeeded": succeeded,
            "errors": {"api": "no_enabled_models"},
            "api_base_url": base_url,
            "endpoint_urls": endpoint_urls,
            "raw_calls": raw_calls,
        }

    raw = _fetch_combined_result(
        base_url=base_url,
        model_names=api_model_order,
        sequence=sequence,
        timeout_seconds=timeout_seconds,
    )
    raw_calls.append(_raw_call_from_fetch(raw, model_names=api_model_order))
    endpoint_urls.append(str(raw.get("url", "")))
    if not raw.get("ok"):
        error_text = str(raw.get("error", "unknown_error"))
        # Some deployments reject semicolon-joined model strings; retry per-model.
        if "404" in error_text or "not_found" in _normalize_label(error_text):
            for spec in enabled_specs:
                ptm_type = str(spec.get("ptm_type", "")).strip()
                api_model = str(spec.get("api_model", ptm_type)).strip() or ptm_type
                score_aliases = spec.get("score_aliases")
                if not isinstance(score_aliases, list) or not score_aliases:
                    if ptm_type == "Phosphoserine_Phosphothreonine":
                        score_aliases = ["Phosphoserine", "Phosphothreonine"]
                    else:
                        score_aliases = [api_model]
                one = _fetch_combined_result(
                    base_url=base_url,
                    model_names=[api_model],
                    sequence=sequence,
                    timeout_seconds=timeout_seconds,
                )
                raw_calls.append(_raw_call_from_fetch(one, model_names=[api_model]))
                endpoint_urls.append(str(one.get("url", "")))
                if not one.get("ok"):
                    errors[ptm_type] = str(one.get("error", "unknown_error"))
                    continue
                results = one.get("results", [])
                positions = _extract_positive_positions(results=results, score_aliases=[str(x) for x in score_aliases])
                predictions_by_type[ptm_type] = positions
                succeeded.append(ptm_type)
        else:
            for ptm_type in attempted:
                errors[ptm_type] = error_text
    else:
        results = raw.get("results", [])
        for spec in enabled_specs:
            ptm_type = str(spec.get("ptm_type", "")).strip()
            api_model = str(spec.get("api_model", ptm_type)).strip() or ptm_type
            score_aliases = spec.get("score_aliases")
            if not isinstance(score_aliases, list) or not score_aliases:
                if ptm_type == "Phosphoserine_Phosphothreonine":
                    score_aliases = ["Phosphoserine", "Phosphothreonine"]
                else:
                    score_aliases = [api_model]
            positions = _extract_positive_positions(results=results, score_aliases=[str(x) for x in score_aliases])
            predictions_by_type[ptm_type] = positions
            succeeded.append(ptm_type)

    status = "ok" if succeeded else "failed"
    if succeeded and not any(predictions_by_type.get(k) for k in succeeded):
        status = "ok_empty"
    return {
        "available": bool(succeeded),
        "status": status,
        "predictions_by_type": predictions_by_type,
        "models_attempted": attempted,
        "models_succeeded": succeeded,
        "errors": errors,
        "api_base_url": base_url,
        "endpoint_urls": endpoint_urls,
        "raw_calls": raw_calls,
    }


def _merge_errors(existing: dict[str, str], incoming: dict[str, str], chunk_idx: int) -> None:
    prefix = f"chunk{int(chunk_idx)}:"
    for key, value in incoming.items():
        k = str(key)
        v = prefix + str(value)
        if k not in existing:
            existing[k] = v
            continue
        if v not in existing[k]:
            existing[k] = f"{existing[k]}|{v}"


def run_musitedeep_api_predictions(
    *,
    sequence: str,
    model_specs: list[dict[str, Any]],
    base_url: str = DEFAULT_MUSITEDEEP_API_BASE_URL,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Run MusiteDeep PTM predictions using HTTPS API.

    MusiteDeep API limits input sequence length (currently 1000 aa). For longer
    proteins we execute chunked requests and remap residue positions.
    """
    seq = str(sequence or "").strip().upper()
    if not seq:
        return {
            "available": False,
            "status": "failed",
            "predictions_by_type": {},
            "models_attempted": [],
            "models_succeeded": [],
            "errors": {"api": "empty_sequence"},
            "api_base_url": base_url,
            "endpoint_urls": [],
            "raw_calls": [],
            "chunking": {"enabled": False, "chunk_size": MUSITEDEEP_MAX_SEQUENCE_LENGTH, "chunk_count": 0},
        }

    if len(seq) <= MUSITEDEEP_MAX_SEQUENCE_LENGTH:
        out = _run_single_sequence_predictions(
            sequence=seq,
            model_specs=model_specs,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        out["chunking"] = {
            "enabled": False,
            "chunk_size": MUSITEDEEP_MAX_SEQUENCE_LENGTH,
            "chunk_count": 1,
            "sequence_length": len(seq),
        }
        return out

    chunk_size = MUSITEDEEP_MAX_SEQUENCE_LENGTH
    merged_positions: dict[str, set[int]] = {}
    merged_errors: dict[str, str] = {}
    merged_urls: list[str] = []
    merged_raw_calls: list[dict[str, Any]] = []
    attempted: list[str] = []
    succeeded: set[str] = set()
    chunk_statuses: list[str] = []

    chunk_count = (len(seq) + chunk_size - 1) // chunk_size
    for chunk_idx, start in enumerate(range(0, len(seq), chunk_size), start=1):
        chunk_seq = seq[start : start + chunk_size]
        chunk = _run_single_sequence_predictions(
            sequence=chunk_seq,
            model_specs=model_specs,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        if not attempted:
            attempted = list(chunk.get("models_attempted", []))
        chunk_statuses.append(str(chunk.get("status", "failed")))
        merged_urls.extend([str(x) for x in chunk.get("endpoint_urls", [])])
        for row in chunk.get("raw_calls", []) or []:
            record = dict(row)
            record["chunk_index"] = int(chunk_idx)
            record["chunk_start"] = int(start + 1)
            record["chunk_end"] = int(start + len(chunk_seq))
            merged_raw_calls.append(record)
        succeeded.update([str(x) for x in chunk.get("models_succeeded", [])])
        _merge_errors(merged_errors, chunk.get("errors", {}) or {}, chunk_idx)

        pred_by_type = chunk.get("predictions_by_type", {}) or {}
        for ptm_type, positions in pred_by_type.items():
            bucket = merged_positions.setdefault(str(ptm_type), set())
            for pos in positions or []:
                try:
                    global_pos = int(pos) + start
                except (TypeError, ValueError):
                    continue
                if global_pos >= 1:
                    bucket.add(global_pos)

    final_predictions = {k: sorted(v) for k, v in merged_positions.items()}
    all_chunks_ok = all(s in {"ok", "ok_empty"} for s in chunk_statuses)
    if not succeeded:
        status = "failed"
    elif all_chunks_ok:
        status = "ok"
        if not any(final_predictions.get(k) for k in final_predictions):
            status = "ok_empty"
    else:
        status = "failed"

    return {
        "available": bool(succeeded) and status != "failed",
        "status": status,
        "predictions_by_type": final_predictions,
        "models_attempted": attempted,
        "models_succeeded": sorted(succeeded),
        "errors": merged_errors,
        "api_base_url": base_url,
        "endpoint_urls": merged_urls,
        "raw_calls": merged_raw_calls,
        "chunking": {
            "enabled": True,
            "chunk_size": chunk_size,
            "chunk_count": chunk_count,
            "sequence_length": len(seq),
        },
    }
