#!/usr/bin/env python3
"""Shared helpers for Site4Drug demo UIs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from site4drug_inference.demo import predict_site

FASTA_HEADER = re.compile(r"^>")


def infer_input_mode(sequence_text: str | None, sequence_file: str | Path | None) -> str:
    """Infer the demo input mode from the provided fields."""
    if str(sequence_text or "").strip():
        return "Paste sequence"
    if str(sequence_file or "").strip():
        return "Upload FASTA"
    return "UniProt lookup only"


def normalize_sequence_text(raw_text: str) -> str:
    """Normalize pasted plain-text or FASTA sequence input."""
    lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Sequence input is empty.")
    seq = "".join(line for line in lines if not FASTA_HEADER.match(line))
    return predict_site.normalize_sequence(seq)


def resolve_demo_input_sequence(
    *,
    uniprot: str,
    input_mode: str | None,
    sequence_text: str,
    sequence_file: str | Path | None,
    allow_online_lookup: bool,
    paste_source_label: str = "sequence_text",
    upload_source_prefix: str = "upload:",
) -> tuple[str, str]:
    """Resolve notebook/Gradio sequence inputs into a normalized sequence."""
    mode = str(input_mode or infer_input_mode(sequence_text, sequence_file)).strip().lower()
    if mode.startswith("paste"):
        return normalize_sequence_text(sequence_text), paste_source_label
    if mode.startswith("upload"):
        if not sequence_file:
            raise ValueError("Please upload a FASTA/TXT file.")
        seq, _ = predict_site.read_sequence_file(Path(sequence_file))
        return predict_site.normalize_sequence(seq), f"{upload_source_prefix}{Path(sequence_file).name}"
    return predict_site.resolve_sequence_from_uniprot(
        uniprot,
        allow_online_lookup=allow_online_lookup,
    )


def ranked_candidates_display_df(payload: dict[str, Any]) -> pd.DataFrame:
    """Build the compact ranked-candidates table used by interactive demos."""
    rows = []
    for row in list(payload.get("ranked_candidates", []) or []):
        rows.append(
            {
                "rank": row.get("rank"),
                "candidate_id": row.get("candidate_id"),
                "mode": row.get("mode"),
                "peptide": row.get("peptide"),
                "position": f"{row.get('start')}-{row.get('end')}",
                "confidence": row.get("confidence"),
                "score": round(float(row.get("confidence_score", 0.0) or 0.0), 3),
                "source": row.get("confidence_source", ""),
                "flags": ", ".join(row.get("flags", []) or []),
                "reason": row.get("reason", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "rank",
            "candidate_id",
            "mode",
            "peptide",
            "position",
            "confidence",
            "score",
            "source",
            "flags",
            "reason",
        ],
    )


def agent_conclusion_md(payload: dict[str, Any]) -> str:
    """Format multi-agent conclusions as markdown bullets."""
    traces = payload.get("agent_traces", {}) or {}
    lines = []
    for key, label in (
        ("bio_agent", "BioAgent"),
        ("chem_agent", "ChemAgent"),
        ("risk_agent", "RiskAgent"),
        ("decision_agent", "DecisionAgent"),
    ):
        trace = traces.get(key, {}) if isinstance(traces, dict) else {}
        parsed = trace.get("parsed", {}) if isinstance(trace, dict) else {}
        summary = str(parsed.get("summary") or parsed.get("rationale") or "").strip()
        lines.append(f"- **{label}**: {summary or 'No conclusion available.'}")
    return "\n".join(lines) if lines else "- No agent conclusions available."


def build_analysis_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Collect the analysis/provenance payload shown in interactive demos."""
    return {
        "self_consistency": payload.get("self_consistency", {}),
        "ptm_summary": payload.get("ptm_summary", {}),
        "motif_summary": payload.get("motif_summary", {}),
        "feature_provenance": payload.get("feature_provenance", {}),
        "raw_api_calls": payload.get("raw_api_calls", {}),
        "iedb_validation": payload.get("iedb_validation", {}),
    }


def resolve_plot_path(result: dict[str, Any], payload: dict[str, Any]) -> str | None:
    """Resolve a generated plot artifact path from a run result."""
    plot_artifacts = payload.get("plot_artifacts", {}) or {}
    plot_png = str(plot_artifacts.get("plot_png", "") or "").strip()
    if plot_png and Path(plot_png).exists():
        return plot_png
    plot_name = str(plot_artifacts.get("plot_png_name", "") or "").strip()
    run_dir = Path(result.get("run_dir", "")) if result.get("run_dir") else None
    if plot_name and run_dir:
        candidate = run_dir / plot_name
        if candidate.exists():
            return str(candidate)
    return None


def build_artifacts_text(result: dict[str, Any], payload: dict[str, Any]) -> str:
    """Format the key artifact paths for display."""
    paths = [
        str(result.get("json_path", "")),
        str(result.get("md_path", "")),
        str(result.get("html_path", "")),
        str((payload.get("raw_api_calls", {}) or {}).get("musitedeep", {}).get("artifact_path", "")),
        str((payload.get("raw_api_calls", {}) or {}).get("scanprosite", {}).get("artifact_path", "")),
        str((payload.get("raw_api_calls", {}) or {}).get("scanprosite", {}).get("meta_artifact_path", "")),
        str(resolve_plot_path(result, payload) or ""),
    ]
    return "\n".join(path for path in paths if path).strip()


def load_report_markdown(result: dict[str, Any]) -> str:
    """Load the generated markdown report if it exists."""
    md_path = Path(result.get("md_path", "")) if result.get("md_path") else None
    if md_path and md_path.exists():
        return md_path.read_text(encoding="utf-8")
    return ""
