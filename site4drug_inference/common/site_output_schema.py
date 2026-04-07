#!/usr/bin/env python3
"""Schema helpers for Site4Drug prediction artifacts."""

from __future__ import annotations

from copy import deepcopy

SCHEMA_VERSION = "3.1"

REQUIRED_KEYS = (
    "schema_version",
    "recommended_modality",
    "modality_confidence",
    "ranked_candidates",
    "candidate_evidence",
    "risk_flags",
    "agent_traces",
    "feature_provenance",
    "token_strategy_used",
    "audit_log",
    "ptm_summary",
    "motif_summary",
    "iedb_validation",
    "orchestrator_trace",
)


def _has_any_keyword(text: str, keywords: set[str]) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


def with_schema_defaults(payload: dict) -> dict:
    """Backfill required schema keys while preserving legacy fields."""
    out = deepcopy(payload)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("recommended_modality", "epitope")
    out.setdefault("modality_confidence", 0.0)
    out.setdefault("ranked_candidates", [])
    out.setdefault("candidate_evidence", [])
    out.setdefault("risk_flags", [])
    out.setdefault("agent_traces", {})
    out.setdefault("feature_provenance", {})
    if isinstance(out.get("feature_provenance"), dict):
        out["feature_provenance"].setdefault("ptm_source", "unknown")
        out["feature_provenance"].setdefault("ptm_rule_version", "unknown")
        out["feature_provenance"].setdefault("motif_source", "unknown")
        out["feature_provenance"].setdefault("motif_library_version", "unknown")
        out["feature_provenance"].setdefault("motif_remote_status", "unknown")
    out.setdefault("token_strategy_used", "unknown")
    out.setdefault("audit_log", {"warnings": [], "events": []})
    out.setdefault("ptm_summary", {})
    if isinstance(out.get("ptm_summary"), dict):
        out["ptm_summary"].setdefault("total_sites", 0)
        out["ptm_summary"].setdefault("counts_by_type", {})
    out.setdefault("motif_summary", {})
    if isinstance(out.get("motif_summary"), dict):
        out["motif_summary"].setdefault("total_hits", 0)
        out["motif_summary"].setdefault("counts_by_motif", {})
    out.setdefault("iedb_validation", {})
    if isinstance(out.get("iedb_validation"), dict):
        out["iedb_validation"].setdefault("enabled", False)
        out["iedb_validation"].setdefault("status", "not_requested")
        out["iedb_validation"].setdefault("source", "none")
        out["iedb_validation"].setdefault("n_reference_spans", 0)
        out["iedb_validation"].setdefault("iou_threshold", 0.3)
        out["iedb_validation"].setdefault("top_k_hit", False)
    out.setdefault("orchestrator_trace", [])
    return out


def validate_site_output(payload: dict, validation_context: dict | None = None) -> list[str]:
    """Validate required fields and return human-readable errors."""
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"missing key: {key}")

    modality = payload.get("recommended_modality")
    if modality not in {"epitope", "pocket", "other"}:
        errors.append("recommended_modality must be one of: epitope, pocket, other")

    conf = payload.get("modality_confidence")
    if not isinstance(conf, (int, float)):
        errors.append("modality_confidence must be numeric")
    elif conf < 0.0 or conf > 1.0:
        errors.append("modality_confidence must be in [0, 1]")

    ranked = payload.get("ranked_candidates", [])
    allow_missing_candidate_id = bool((validation_context or {}).get("allow_missing_candidate_id", False))
    if not isinstance(ranked, list):
        errors.append("ranked_candidates must be a list")
    else:
        for i, item in enumerate(ranked):
            if not isinstance(item, dict):
                errors.append(f"ranked_candidates[{i}] must be an object")
                continue
            required_keys = ("rank", "start", "end", "peptide")
            if not allow_missing_candidate_id:
                required_keys = ("rank", "candidate_id", "start", "end", "peptide")
            for required in required_keys:
                if required not in item:
                    errors.append(f"ranked_candidates[{i}] missing {required}")

    if not isinstance(payload.get("orchestrator_trace", []), list):
        errors.append("orchestrator_trace must be a list")

    if not isinstance(payload.get("iedb_validation", {}), dict):
        errors.append("iedb_validation must be an object")

    fp = payload.get("feature_provenance", {})
    if not isinstance(fp, dict):
        errors.append("feature_provenance must be an object")
    else:
        for key in ("ptm_source", "ptm_rule_version", "motif_source", "motif_library_version", "motif_remote_status"):
            if key not in fp:
                errors.append(f"feature_provenance missing {key}")

    context = validation_context or {}
    ptm_sites = context.get("ptm_sites", [])
    motif_hits = context.get("motif_hits", [])

    reasons_blob = " ".join(
        str(item.get("reason", "")) for item in payload.get("ranked_candidates", []) if isinstance(item, dict)
    )
    evidence_blob = " ".join(
        " ".join(str(x) for x in row.get("evidence", []))
        for row in payload.get("candidate_evidence", [])
        if isinstance(row, dict)
    )
    combined = f"{reasons_blob} {evidence_blob}"

    if ptm_sites:
        ptm_keywords = {
            "ptm",
            "glyco",
            "glycosyl",
            "phospho",
            "ubiquit",
            "acetyl",
            "methyl",
            "hydroxy",
            "pyrrolidone",
        }
        if not _has_any_keyword(combined, ptm_keywords):
            errors.append("missing_ptm_evidence")

    if motif_hits:
        motif_keywords = {"motif", "zinc", "nls", "dna", "p-loop", "zipper", "helix"}
        if not _has_any_keyword(combined, motif_keywords):
            errors.append("missing_motif_evidence")

    return errors
