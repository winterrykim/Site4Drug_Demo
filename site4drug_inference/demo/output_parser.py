#!/usr/bin/env python3
"""Robust JSON parsing utilities for Site4Drug LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Callable

from site4drug_inference.common.site_output_schema import validate_site_output, with_schema_defaults

JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
RANK_BLOCK_PATTERN = re.compile(r"\*\*Rank\s*(\d+)\*\*(.*?)(?=(?:\*\*Rank\s*\d+\*\*)|\Z)", re.DOTALL | re.IGNORECASE)
POSITION_PATTERN = re.compile(r"Position:\s*(\d+)\s*-\s*(\d+)", re.IGNORECASE)
EPITOPE_PATTERN = re.compile(r"Epitope:\s*([A-Za-z0-9_]+)", re.IGNORECASE)
CONFIDENCE_PATTERN = re.compile(r"Confidence:\s*(High|Moderate|Low)", re.IGNORECASE)
RISK_PATTERN = re.compile(r"Risk:\s*([^\n\r]+)", re.IGNORECASE)
CANDIDATE_BLOCK_PATTERN = re.compile(
    r"Candidate\s*(\d+)\s*:\s*([A-Za-z0-9_]+)\s*\((\d+)\s*-\s*(\d+)[^)]*\)",
    re.IGNORECASE,
)
CONTROL_TOKEN_PATTERN = re.compile(r"<\|(?:im_end|im_start|endoftext|eot_id)\|>")
LIKELY_OBJECT_KEY_PATTERN = re.compile(
    r'"(?:recommended_modality|ranked_candidates|agent|modality_votes|candidate_adjustments|ranking|audit_log)"\s*:',
    re.IGNORECASE,
)
RECOMMENDED_MODALITY_PATTERN = re.compile(r'"recommended_modality"\s*:\s*"(epitope|pocket|other)"', re.IGNORECASE)
MODALITY_CONFIDENCE_PATTERN = re.compile(r'"modality_confidence"\s*:\s*([0-9]*\.?[0-9]+)')
START_END_PEPTIDE_PATTERN = re.compile(
    r'"start"\s*:\s*(\d+)\s*[,}]'
    r'.{0,240}?'
    r'"end"\s*:\s*(\d+)\s*[,}]'
    r'.{0,240}?'
    r'"peptide"\s*:\s*"([A-Za-z]+)"',
    re.IGNORECASE | re.DOTALL,
)
PEPTIDE_START_END_PATTERN = re.compile(
    r'"peptide"\s*:\s*"([A-Za-z]+)"'
    r'.{0,240}?'
    r'"start"\s*:\s*(\d+)\s*[,}]'
    r'.{0,240}?'
    r'"end"\s*:\s*(\d+)\s*[,}]',
    re.IGNORECASE | re.DOTALL,
)
PAYLOAD_ANCHOR_KEYS = {
    "recommended_modality",
    "ranked_candidates",
    "agent",
    "modality_votes",
    "candidate_adjustments",
    "ranking",
    "audit_log",
}


def _normalize_model_text(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.replace("\ufeff", "")
    normalized = CONTROL_TOKEN_PATTERN.sub("", normalized)
    return normalized.strip()


def _extract_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start_idx: int | None = None
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                objects.append(text[start_idx : idx + 1])
                start_idx = None

    return objects


def _looks_like_object_body_without_braces(text: str) -> bool:
    compact = text.lstrip()
    if not compact.startswith('"'):
        return False
    return bool(LIKELY_OBJECT_KEY_PATTERN.search(compact))


def _payload_score(obj: dict) -> tuple[int, int, int]:
    keys = {str(k) for k in obj.keys()}
    anchor_hits = len(keys.intersection(PAYLOAD_ANCHOR_KEYS))
    semantic_score = 0
    if "recommended_modality" in keys:
        semantic_score += 4
    if "ranked_candidates" in keys:
        semantic_score += 4
    if {"agent", "modality_votes", "candidate_adjustments"}.issubset(keys):
        semantic_score += 6
    if {"recommended_modality", "ranking"}.issubset(keys):
        semantic_score += 6
    # Penalize common nested fragments so we keep searching for full objects.
    if keys.issubset({"epitope", "pocket", "other"}):
        semantic_score -= 6
    if {"rank", "candidate_id", "reason"}.issubset(keys) and len(keys) <= 6:
        semantic_score -= 5
    return (anchor_hits, semantic_score, len(keys))


def _candidate_json_strings(text: str) -> list[str]:
    candidates: list[str] = []
    normalized = _normalize_model_text(text)

    for match in JSON_BLOCK_PATTERN.finditer(text):
        candidates.append(match.group(1).strip())
    if _looks_like_object_body_without_braces(normalized):
        candidates.append("{" + normalized)
    candidates.extend(_extract_balanced_json_objects(normalized))

    # Deduplicate while preserving order.
    unique = []
    seen = set()
    for cand in candidates:
        if cand not in seen:
            unique.append(cand)
            seen.add(cand)
    return unique


def parse_json_object(text: str) -> tuple[dict | None, str | None]:
    """Parse the first valid JSON object found in text."""
    normalized = _normalize_model_text(text)

    # Fast path: entire text is JSON.
    try:
        obj = json.loads(normalized)
        if isinstance(obj, dict):
            return obj, None
    except json.JSONDecodeError:
        pass

    parsed_candidates: list[tuple[tuple[int, int, int], dict]] = []
    for candidate in _candidate_json_strings(normalized):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                parsed_candidates.append((_payload_score(obj), obj))
        except json.JSONDecodeError:
            continue

    if parsed_candidates:
        anchored = [item for item in parsed_candidates if item[0][0] > 0]
        if anchored:
            anchored.sort(key=lambda item: item[0], reverse=True)
            return anchored[0][1], None

    return None, "no_valid_json_object_found"


def _infer_modality(text: str) -> str:
    low = text.lower()
    has_epitope = "epitope" in low
    has_pocket = "pocket" in low
    if has_epitope and not has_pocket:
        return "epitope"
    if has_pocket and not has_epitope:
        return "pocket"
    if has_epitope and has_pocket:
        return "other"
    return "epitope"


def _clean_peptide(raw: str) -> str:
    pep = re.sub(r"[^A-Za-z]", "", raw or "").upper()
    return pep


def _coerce_int(raw: object) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _coerce_float(raw: object) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _normalize_confidence(raw: str | None) -> str:
    if not raw:
        return "Moderate"
    text = str(raw).strip().lower()
    if text.startswith("high"):
        return "High"
    if text.startswith("low"):
        return "Low"
    return "Moderate"


def _candidate_rows_from_rank_blocks(text: str) -> list[dict]:
    rows: list[dict] = []
    for match in RANK_BLOCK_PATTERN.finditer(text):
        rank = int(match.group(1))
        block = match.group(2)
        pos_m = POSITION_PATTERN.search(block)
        epi_m = EPITOPE_PATTERN.search(block)
        conf_m = CONFIDENCE_PATTERN.search(block)
        if not pos_m:
            continue
        start = int(pos_m.group(1))
        end = int(pos_m.group(2))
        if end < start:
            continue
        peptide = _clean_peptide(epi_m.group(1) if epi_m else "")
        if not peptide:
            peptide = ""
        rows.append(
            {
                "rank": rank,
                "candidate_id": f"L_R{rank:04d}",
                "start": start,
                "end": end,
                "peptide": peptide,
                "confidence": _normalize_confidence(conf_m.group(1) if conf_m else None),
                "flags": [],
                "reason": "Recovered from legacy text output (constraint-first PTM/motif context retained).",
            }
        )
    rows.sort(key=lambda r: r.get("rank", 9999))
    return rows


def _candidate_rows_from_candidate_blocks(text: str) -> list[dict]:
    matches = list(CANDIDATE_BLOCK_PATTERN.finditer(text))
    rows: list[dict] = []
    for idx, match in enumerate(matches, start=1):
        block_start = match.start()
        block_end = matches[idx].start() if idx < len(matches) else len(text)
        block = text[block_start:block_end]
        start = int(match.group(3))
        end = int(match.group(4))
        if end < start:
            continue
        peptide = _clean_peptide(match.group(2))
        conf_m = CONFIDENCE_PATTERN.search(block)
        risk_m = RISK_PATTERN.search(block)
        flags: list[str] = []
        if risk_m:
            risk_text = risk_m.group(1).strip()
            if risk_text and risk_text.lower() not in {"none", "n/a"}:
                flags = [tok.strip() for tok in risk_text.split(",") if tok.strip()]
        rows.append(
            {
                "rank": idx,
                "candidate_id": f"L_C{idx:04d}",
                "start": start,
                "end": end,
                "peptide": peptide,
                "confidence": _normalize_confidence(conf_m.group(1) if conf_m else None),
                "flags": flags,
                "reason": "Recovered from legacy candidate block (constraint-first PTM/motif context retained).",
            }
        )
    return rows


def _extract_mode_and_confidence(text: str) -> tuple[str, float]:
    mode_match = RECOMMENDED_MODALITY_PATTERN.search(text)
    conf_match = MODALITY_CONFIDENCE_PATTERN.search(text)
    mode = mode_match.group(1).lower() if mode_match else _infer_modality(text)
    conf = 0.45
    if conf_match:
        try:
            conf = float(conf_match.group(1))
        except ValueError:
            conf = 0.45
    conf = min(max(conf, 0.0), 1.0)
    return mode, conf


def _candidate_rows_from_json_fragments(text: str) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    idx = 1

    for fragment in _extract_balanced_json_objects(_normalize_model_text(text)):
        try:
            obj = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if not {"start", "end", "peptide"}.issubset(obj.keys()):
            continue

        start = int(obj.get("start", -1))
        end = int(obj.get("end", -1))
        peptide = _clean_peptide(str(obj.get("peptide", "")))
        if start <= 0 or end < start or not peptide:
            continue

        key = (start, end, peptide)
        if key in seen:
            continue
        seen.add(key)

        flags = obj.get("flags", [])
        if not isinstance(flags, list):
            flags = []
        raw_conf = obj.get("confidence")
        rows.append(
            {
                "rank": int(obj.get("rank", idx)),
                "candidate_id": str(obj.get("candidate_id", f"K_C{idx:04d}")),
                "start": start,
                "end": end,
                "peptide": peptide,
                "confidence": _normalize_confidence(str(raw_conf) if raw_conf is not None else None),
                "flags": [str(f) for f in flags],
                "reason": str(obj.get("reason", "Recovered from partial JSON fragment.")),
            }
        )
        idx += 1

    rows.sort(key=lambda r: r.get("rank", 9999))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        if not row.get("candidate_id"):
            row["candidate_id"] = f"K_C{i:04d}"
    return rows


def _candidate_rows_from_keyword_patterns(text: str) -> list[dict]:
    rows: list[dict] = []
    seen = set()

    def _add(start: int, end: int, pep: str) -> None:
        if start <= 0 or end < start:
            return
        peptide = _clean_peptide(pep)
        if not peptide:
            return
        key = (start, end, peptide)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "rank": len(rows) + 1,
                "candidate_id": f"K_R{len(rows) + 1:04d}",
                "start": start,
                "end": end,
                "peptide": peptide,
                "confidence": "Moderate",
                "flags": [],
                "reason": "Recovered from keyword pattern in malformed JSON/text output.",
            }
        )

    for m in START_END_PEPTIDE_PATTERN.finditer(text):
        _add(int(m.group(1)), int(m.group(2)), m.group(3))
    for m in PEPTIDE_START_END_PATTERN.finditer(text):
        _add(int(m.group(2)), int(m.group(3)), m.group(1))
    return rows


def _build_compact_proposal_payload(
    *,
    mode: str,
    conf: float,
    ranked: list[dict],
    parsed_from: str,
    warning: str | None = None,
) -> dict:
    risk_flags: set[str] = set()
    normalized_rows: list[dict] = []
    for idx, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = idx
        peptide = _clean_peptide(str(item.get("peptide", "")))
        if not peptide:
            continue
        item["peptide"] = peptide
        flags = item.get("flags", [])
        if not isinstance(flags, list):
            flags = []
        item["flags"] = [str(flag) for flag in flags]
        risk_flags.update(item["flags"])
        normalized_rows.append(item)
    warnings = [warning] if warning else []
    return with_schema_defaults(
        {
            "recommended_modality": mode,
            "modality_confidence": conf,
            "ranked_candidates": normalized_rows,
            "candidate_evidence": [],
            "risk_flags": sorted(risk_flags),
            "agent_traces": {},
            "feature_provenance": {},
            "token_strategy_used": parsed_from,
            "audit_log": {
                "warnings": warnings,
                "events": [],
            },
            "_parsed_from": parsed_from,
        }
    )


def _normalize_proposal_payload(payload: dict | None) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    if payload is None:
        return None, ["payload_missing"]
    if not isinstance(payload, dict):
        return None, ["payload_not_object"]

    mode = str(payload.get("recommended_modality", "") or "").strip().lower()
    if mode not in {"epitope", "pocket", "other"}:
        errors.append("invalid_recommended_modality")

    conf = _coerce_float(payload.get("modality_confidence"))
    if conf is None:
        errors.append("modality_confidence_not_numeric")
    else:
        conf = min(max(conf, 0.0), 1.0)

    ranked = payload.get("ranked_candidates")
    if not isinstance(ranked, list):
        errors.append("ranked_candidates_not_list")
        ranked = []

    normalized_rows: list[tuple[int, dict]] = []
    seen: set[tuple[str, int, int, str]] = set()
    row_warnings: list[str] = []
    for idx, row in enumerate(ranked, start=1):
        if not isinstance(row, dict):
            row_warnings.append(f"proposal_row_{idx}_not_object")
            continue

        start = _coerce_int(row.get("start"))
        end = _coerce_int(row.get("end"))
        peptide = _clean_peptide(str(row.get("peptide", "")))
        if start is None or end is None or start <= 0 or end < start or not peptide:
            row_warnings.append(f"proposal_row_{idx}_missing_span_fields")
            continue

        raw_mode = str(row.get("mode", "") or "").strip().lower()
        dedupe_mode = raw_mode if raw_mode in {"epitope", "pocket", "other"} else ""
        dedupe_key = (dedupe_mode, start, end, peptide)
        if dedupe_key in seen:
            row_warnings.append(f"proposal_row_{idx}_duplicate")
            continue
        seen.add(dedupe_key)

        if len(peptide) != end - start + 1:
            row_warnings.append(f"proposal_row_{idx}_peptide_span_length_mismatch")

        normalized = {
            "rank": _coerce_int(row.get("rank")) or idx,
            "start": start,
            "end": end,
            "peptide": peptide,
            "reason": str(row.get("reason", "") or "").strip(),
        }
        if row.get("candidate_id") is not None:
            normalized["candidate_id"] = str(row.get("candidate_id"))
        if raw_mode:
            normalized["mode"] = raw_mode
        if row.get("confidence") is not None:
            normalized["confidence"] = str(row.get("confidence"))
        if row.get("confidence_score") is not None:
            normalized["confidence_score"] = row.get("confidence_score")
        if row.get("confidence_reason") is not None:
            normalized["confidence_reason"] = str(row.get("confidence_reason"))
        flags = row.get("flags", [])
        if isinstance(flags, list):
            normalized["flags"] = [str(flag) for flag in flags]
        normalized_rows.append((normalized["rank"], normalized))

    if not normalized_rows:
        errors.append("ranked_candidates_empty")

    if errors:
        return None, errors

    normalized_rows.sort(key=lambda item: (item[0], item[1].get("start", 0), item[1].get("end", 0)))
    payload_out = _build_compact_proposal_payload(
        mode=mode,
        conf=conf if conf is not None else 0.0,
        ranked=[row for _, row in normalized_rows],
        parsed_from="proposal_json",
        warning="proposal_row_cleanup_applied" if row_warnings else None,
    )
    if row_warnings:
        payload_out.setdefault("audit_log", {}).setdefault("warnings", []).extend(row_warnings[:20])
    return payload_out, []


def parse_keyword_site_output(text: str, *, compact: bool = False) -> dict | None:
    """Best-effort recovery from malformed/truncated JSON-like outputs."""
    normalized = _normalize_model_text(text)
    ranked = _candidate_rows_from_json_fragments(normalized)
    if not ranked:
        ranked = _candidate_rows_from_keyword_patterns(normalized)
    if not ranked:
        return None

    mode, conf = _extract_mode_and_confidence(normalized)
    if compact:
        return _build_compact_proposal_payload(
            mode=mode,
            conf=conf,
            ranked=ranked,
            parsed_from="keyword_recovery",
            warning="keyword_recovery_used",
        )

    risk_flags: set[str] = set()
    candidate_evidence: list[dict] = []
    for row in ranked:
        row_flags = [str(flag) for flag in row.get("flags", [])]
        risk_flags.update(row_flags)
        candidate_evidence.append(
            {
                "candidate_id": row["candidate_id"],
                "evidence": [
                    "keyword_recovery",
                    "partial_json_fragment",
                ],
            }
        )

    payload = with_schema_defaults(
        {
            "recommended_modality": mode,
            "modality_confidence": conf,
            "ranked_candidates": ranked,
            "candidate_evidence": candidate_evidence,
            "risk_flags": sorted(risk_flags),
            "agent_traces": {},
            "feature_provenance": {},
            "token_strategy_used": "keyword_recovery",
            "audit_log": {
                "warnings": ["keyword_recovery_used"],
                "events": [],
            },
            "_parsed_from": "keyword_recovery",
        }
    )
    return payload


def parse_legacy_site_output(text: str, *, compact: bool = False) -> dict | None:
    """Attempt to recover a Site4Drug payload from legacy non-JSON text."""
    ranked = _candidate_rows_from_rank_blocks(text)
    if not ranked:
        ranked = _candidate_rows_from_candidate_blocks(text)
    if not ranked:
        return None

    # Deduplicate by span while preserving order.
    deduped: list[dict] = []
    seen = set()
    for row in ranked:
        key = (row.get("start"), row.get("end"), row.get("peptide"))
        if key in seen:
            continue
        seen.add(key)
        row["rank"] = len(deduped) + 1
        deduped.append(row)

    if compact:
        return _build_compact_proposal_payload(
            mode=_infer_modality(text),
            conf=0.45,
            ranked=deduped,
            parsed_from="legacy_text",
            warning="legacy_text_output_recovered",
        )

    risk_flags: set[str] = set()
    candidate_evidence: list[dict] = []
    for row in deduped:
        row_flags = [str(flag) for flag in row.get("flags", [])]
        risk_flags.update(row_flags)
        candidate_evidence.append(
            {
                "candidate_id": row["candidate_id"],
                "evidence": [
                    "legacy_text_recovery",
                    "ptm_motif_context_available",
                ],
            }
        )

    payload = with_schema_defaults(
        {
            "recommended_modality": _infer_modality(text),
            "modality_confidence": 0.45,
            "ranked_candidates": deduped,
            "candidate_evidence": candidate_evidence,
            "risk_flags": sorted(risk_flags),
            "agent_traces": {},
            "feature_provenance": {},
            "token_strategy_used": "legacy_text_recovery",
            "audit_log": {
                "warnings": ["legacy_text_output_recovered"],
                "events": [],
            },
            "_parsed_from": "legacy_text",
        }
    )
    return payload


def parse_proposal_output(text: str, validation_context: dict | None = None) -> tuple[dict | None, list[str]]:
    """Parse compact LLM proposal payloads before final enrichment/validation."""
    obj, parse_error = parse_json_object(text)
    collected_errors: list[str] = []
    if obj is not None:
        proposal_obj, proposal_errors = _normalize_proposal_payload(obj)
        if proposal_obj is not None:
            return proposal_obj, []
        collected_errors.extend(proposal_errors)

    legacy_obj = parse_legacy_site_output(text, compact=True)
    if legacy_obj is not None and legacy_obj.get("ranked_candidates"):
        return legacy_obj, []

    keyword_obj = parse_keyword_site_output(text, compact=True)
    if keyword_obj is not None and keyword_obj.get("ranked_candidates"):
        return keyword_obj, []

    if parse_error:
        collected_errors.append(parse_error)
    if not collected_errors:
        collected_errors.append("parse_failed")
    return None, collected_errors


def parse_site_output(
    text: str,
    validation_context: dict | None = None,
    parse_mode: str = "final",
) -> tuple[dict | None, list[str]]:
    """Parse and validate Site4Drug output object."""
    if parse_mode == "proposal":
        return parse_proposal_output(text, validation_context=validation_context)

    obj, parse_error = parse_json_object(text)
    if obj is not None:
        obj = with_schema_defaults(obj)
        errors = validate_site_output(obj, validation_context=validation_context)
        if not errors:
            return obj, []
        return None, errors

    # Compatibility fallback for checkpoints still trained to legacy prose targets.
    legacy_obj = parse_legacy_site_output(text)
    if legacy_obj is not None:
        legacy_errors = validate_site_output(legacy_obj, validation_context=validation_context)
        if not legacy_errors:
            return legacy_obj, []
        return None, legacy_errors

    keyword_obj = parse_keyword_site_output(text)
    if keyword_obj is not None:
        keyword_errors = validate_site_output(keyword_obj, validation_context=validation_context)
        if not keyword_errors:
            return keyword_obj, []
        return None, keyword_errors

    return None, [parse_error or "parse_failed"]


def parse_with_single_repair(
    raw_text: str,
    repair_fn: Callable[[str], str] | None = None,
    validation_context: dict | None = None,
    max_repairs: int = 1,
    parse_mode: str = "final",
) -> tuple[dict | None, dict]:
    """Try strict parse, then optionally retry with repair callback up to max_repairs."""
    parsed, errors = parse_site_output(
        raw_text,
        validation_context=validation_context,
        parse_mode=parse_mode,
    )
    if parsed is not None:
        parsed_from = parsed.get("_parsed_from")
        if parsed_from == "legacy_text":
            status = "legacy_text_parsed"
        elif parsed_from == "keyword_recovery":
            status = "keyword_recovered"
        else:
            status = "ok"
        return parsed, {"parser_status": status, "parser_errors": [], "parse_mode": parse_mode}

    if repair_fn is None:
        return None, {"parser_status": "failed", "parser_errors": errors, "parse_mode": parse_mode}

    max_repairs = max(0, int(max_repairs))
    all_errors = list(errors)
    last_errors = list(errors)
    for repair_idx in range(1, max_repairs + 1):
        if parse_mode == "proposal":
            target_count = _coerce_int((validation_context or {}).get("proposal_target_count"))
            count_rule = (
                f"Return up to {target_count} ranked_candidates.\n"
                if target_count is not None and target_count > 0
                else "Return ranked_candidates only for recoverable rows.\n"
            )
            schema_hint = (
                '{\n'
                '  "recommended_modality": "epitope|pocket|other",\n'
                '  "modality_confidence": 0.0,\n'
                '  "ranked_candidates": [\n'
                '    {\n'
                '      "rank": 1,\n'
                '      "start": 1,\n'
                '      "end": 15,\n'
                '      "peptide": "AAAA",\n'
                '      "mode": "epitope|pocket|other",\n'
                '      "confidence_score": 0.0,\n'
                '      "reason": "Short PTM/motif-aware rationale."\n'
                '    }\n'
                "  ]\n"
                "}\n"
                f"{count_rule}"
                "candidate_id is optional and will be assigned later.\n"
            )
        else:
            schema_hint = (
                '{\n'
                '  "recommended_modality": "epitope|pocket|other",\n'
                '  "modality_confidence": 0.0,\n'
                '  "ranked_candidates": [\n'
                '    {\n'
                '      "rank": 1,\n'
                '      "candidate_id": "C0001",\n'
                '      "start": 1,\n'
                '      "end": 15,\n'
                '      "peptide": "AAAA",\n'
                '      "mode": "epitope|pocket|other",\n'
                '      "confidence": "High|Moderate|Low",\n'
                '      "flags": [],\n'
                '      "reason": "..."\n'
                '    }\n'
                "  ],\n"
                '  "candidate_evidence": [{"candidate_id":"C0001","evidence":["..."]}],\n'
                '  "risk_flags": [],\n'
                '  "agent_traces": {},\n'
                '  "feature_provenance": {\n'
                '    "ptm_source": "...",\n'
                '    "ptm_rule_version": "...",\n'
                '    "motif_source": "...",\n'
                '    "motif_library_version": "...",\n'
                '    "motif_remote_status": "..."\n'
                "  },\n"
                '  "ptm_summary": {"total_sites": 0, "counts_by_type": {}},\n'
                '  "motif_summary": {"total_hits": 0, "counts_by_motif": {}},\n'
                '  "iedb_validation": {"enabled": false, "status": "not_requested", "source": "none", "n_reference_spans": 0, "iou_threshold": 0.3, "top_k_hit": false},\n'
                '  "orchestrator_trace": [],\n'
                '  "token_strategy_used": "...",\n'
                '  "audit_log": {"warnings": [], "events": []}\n'
                "}"
            )
        ptm_sites = list((validation_context or {}).get("ptm_sites", []) or [])
        motif_hits = list((validation_context or {}).get("motif_hits", []) or [])
        context_lines: list[str] = []
        if ptm_sites:
            ptm_preview = ", ".join(
                f"{site.get('ptm_type', 'unknown')}@{site.get('position', '?')}"
                for site in ptm_sites[:8]
                if isinstance(site, dict)
            )
            context_lines.append(
                f"PTM context is present ({len(ptm_sites)} sites). Mention PTM evidence/caveats when relevant. Sample: {ptm_preview}"
            )
        if motif_hits:
            motif_preview = ", ".join(
                f"{hit.get('motif_name', 'motif')}@{hit.get('start', '?')}-{hit.get('end', '?')}"
                for hit in motif_hits[:8]
                if isinstance(hit, dict)
            )
            context_lines.append(
                f"Motif context is present ({len(motif_hits)} hits). Mention motif evidence/caveats when relevant. Sample: {motif_preview}"
            )
        context_block = "\n".join(context_lines)
        repair_prompt = (
            "Your previous response failed JSON schema validation.\n"
            "Return ONLY valid JSON object matching the required schema.\n"
            "Do not include markdown/code fences, comments, or extra text before/after JSON.\n"
            "Output must start with '{' and end with '}'.\n"
            f"Current issues to fix: {last_errors}\n"
            f"{context_block}\n" if context_block else ""
            "Expected schema template:\n"
            f"{schema_hint}\n"
        )
        repaired_text = repair_fn(repair_prompt)
        repaired, repaired_errors = parse_site_output(
            repaired_text,
            validation_context=validation_context,
            parse_mode=parse_mode,
        )
        if repaired is not None:
            repaired_from = repaired.get("_parsed_from")
            if repaired_from == "legacy_text":
                status = "repaired_legacy_text"
            elif repaired_from == "keyword_recovery":
                status = "repaired_keyword_recovery"
            else:
                status = "repaired"
            return repaired, {
                "parser_status": status,
                "parser_errors": all_errors,
                "repair_attempted": True,
                "repair_attempts": repair_idx,
                "parse_mode": parse_mode,
            }
        last_errors = list(repaired_errors)
        all_errors.extend(repaired_errors)

    return None, {
        "parser_status": "failed_after_repair",
        "parser_errors": all_errors,
        "repair_attempted": max_repairs > 0,
        "repair_attempts": max_repairs,
        "parse_mode": parse_mode,
    }
