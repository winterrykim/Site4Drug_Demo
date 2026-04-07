#!/usr/bin/env python3
"""Motif feature extraction helpers for Site4Drug."""

from __future__ import annotations

from site4drug_inference.common.motif_remote import scanprosite_lookup

def scan_motifs(
    sequence: str,
    source: str = "remote",
    enabled: bool = True,
    timeout: int = 8,
) -> tuple[list[dict], dict]:
    """Scan sequence motifs via remote ScanProsite.

    Returns (hits, metadata).
    """
    seq = (sequence or "").strip().upper()
    if not enabled:
        return [], {"motif_source": "disabled", "motif_library_version": "none", "motif_remote_status": "disabled"}

    lib_version = "scanprosite_biopython_v1"
    source_norm = str(source or "remote").strip().lower()
    if source_norm == "local":
        # Local toy motif snapshots were removed for publication reproducibility.
        return [], {
            "motif_source": "local_disabled",
            "motif_library_version": lib_version,
            "motif_remote_status": "local_disabled",
        }
    if source_norm not in {"remote", "auto"}:
        source_norm = "remote"

    remote_hits, remote_meta = scanprosite_lookup(seq, timeout=timeout)
    remote_status = str(remote_meta.get("status", "unknown"))
    remote_error = str(remote_meta.get("error", "") or "").strip()
    remote_warning = str(remote_meta.get("warning", "") or "").strip()
    remote_raw_xml = str(remote_meta.get("raw_xml", "") or "")
    if remote_hits:
        out_meta = {
            "motif_source": "remote",
            "motif_library_version": lib_version,
            "motif_remote_status": remote_status,
            "motif_remote_raw_xml": remote_raw_xml,
        }
        if remote_warning:
            out_meta["motif_remote_warning"] = remote_warning
        return remote_hits, out_meta

    empty_status = remote_status
    if empty_status == "remote_ok":
        empty_status = "remote_ok_no_hits"

    out_meta = {
        "motif_source": "remote_unavailable" if source_norm == "auto" else "remote",
        "motif_library_version": lib_version,
        "motif_remote_status": empty_status,
        "motif_remote_raw_xml": remote_raw_xml,
    }
    if remote_error:
        out_meta["motif_remote_error"] = remote_error
    if remote_warning:
        out_meta["motif_remote_warning"] = remote_warning
    return [], out_meta


def summarize_motif_hits(motif_hits: list[dict]) -> dict:
    by_name: dict[str, int] = {}
    for hit in motif_hits:
        name = str(hit.get("motif_name", "Unknown_motif"))
        by_name[name] = by_name.get(name, 0) + 1
    return {
        "total_hits": len(motif_hits),
        "counts_by_motif": by_name,
    }
