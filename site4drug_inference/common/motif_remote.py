#!/usr/bin/env python3
"""Optional remote motif lookup helpers (best-effort, non-blocking)."""

from __future__ import annotations

import socket
from typing import Any
from xml.etree import ElementTree as ET


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def scanprosite_lookup(
    sequence: str,
    timeout: int = 8,
    mirror: str = "https://prosite.expasy.org",
) -> tuple[list[dict], dict]:
    """Attempt remote motif lookup via Biopython ScanProsite.

    This function is intentionally defensive: any network/parsing error returns
    an empty hit list with a status payload instead of raising.
    """
    seq = (sequence or "").strip().upper()
    if not seq:
        return [], {"status": "empty_sequence"}

    try:
        from Bio.ExPASy import ScanProsite  # type: ignore
    except Exception as exc:
        return [], {"status": "remote_unavailable_biopython", "error": str(exc)}

    try:
        # Biopython ScanProsite does not expose a dedicated timeout argument.
        # Enforce timeout via the process-default socket timeout to avoid
        # indefinite hangs on remote calls.
        try:
            effective_timeout = max(int(timeout), 1)
        except (TypeError, ValueError):
            effective_timeout = 8
        previous_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(effective_timeout)
            handle = ScanProsite.scan(seq=seq, mirror=mirror, output="xml")
            raw = handle.read()
        finally:
            socket.setdefaulttimeout(previous_timeout)
    except Exception as exc:
        return [], {"status": "remote_failed", "error": str(exc), "raw_xml": ""}

    xml_text: str
    if isinstance(raw, bytes):
        xml_text = raw.decode("utf-8", errors="replace")
    else:
        xml_text = str(raw or "")
    if not xml_text.strip():
        return [], {"status": "remote_failed", "error": "empty_response", "raw_xml": ""}

    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        return [], {"status": "remote_failed", "error": f"xml_parse_error:{exc}", "raw_xml": xml_text}

    warning = ""
    warning_node = root.find(".//{*}warning")
    if warning_node is not None and warning_node.text:
        warning = warning_node.text.strip()
    error_node = root.find(".//{*}error")
    if error_node is not None and error_node.text:
        return [], {"status": "remote_failed", "error": error_node.text.strip(), "raw_xml": xml_text}

    hits: list[dict] = []
    for item in root.findall(".//{*}match"):
        start_node = item.find("{*}start")
        stop_node = item.find("{*}stop")
        signature_ac_node = item.find("{*}signature_ac")
        signature_name_node = item.find("{*}signature_name")
        level_tag_node = item.find("{*}level_tag")
        level_node = item.find("{*}level")
        sequence_id_node = item.find("{*}sequence_id")

        start = _coerce_int(start_node.text if start_node is not None else None)
        end = _coerce_int(stop_node.text if stop_node is not None else None)
        if start is None:
            continue
        if end is None:
            end = start
        signature_ac = str(signature_ac_node.text if signature_ac_node is not None else "").strip()
        signature_name = str(signature_name_node.text if signature_name_node is not None else "").strip()
        level_tag = str(level_tag_node.text if level_tag_node is not None else "").strip()
        level = str(level_node.text if level_node is not None else "").strip()
        sequence_id = str(sequence_id_node.text if sequence_id_node is not None else "").strip()
        if signature_name:
            motif_name = signature_name
        elif signature_ac:
            motif_name = signature_ac
        else:
            motif_name = "Remote_motif"
        hit = {
            "motif_name": motif_name,
            "pattern_id": signature_ac or "REMOTE",
            "start": start,
            "end": end,
            "description": level_tag or level,
            "evidence_source": "remote_scanprosite_biopython",
        }
        if sequence_id:
            hit["sequence_id"] = sequence_id
        hits.append(hit)

    status = "remote_ok_warning" if warning else "remote_ok"
    if not hits and status == "remote_ok":
        status = "remote_ok_no_hits"

    matchset = root.find(".//{*}matchset")
    n_match = None
    n_seq = None
    if matchset is not None:
        n_match = _coerce_int(matchset.attrib.get("n_match"))
        n_seq = _coerce_int(matchset.attrib.get("n_seq"))
    return hits, {
        "status": status,
        "n_hits": len(hits),
        "n_match": n_match if n_match is not None else len(hits),
        "n_seq": n_seq if n_seq is not None else 1,
        "warning": warning,
        "raw_xml": xml_text,
    }
