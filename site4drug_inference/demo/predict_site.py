#!/usr/bin/env python3
"""Run Site4Drug modality-aware site prediction and save auditable artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from site4drug_inference.common.constraint_features import (
    _heuristic_score,
    _motif_hits_in_region,
    _ptm_penalty,
    _risk_flags_for_region,
    aa_composition,
    build_candidate_features,
    count_in_region,
    region_hydropathy,
    region_overlaps,
    slice_sequence_with_overlap,
)
from site4drug_inference.common.env_utils import ensure_openrouter_api_key, ensure_tinker_api_key
from site4drug_inference.common.model_defaults import BASE_MODEL, DEFAULT_CHECKPOINT
from site4drug_inference.common.musitedeep_api import DEFAULT_MUSITEDEEP_API_BASE_URL as MUSITEDEEP_API_BASE_URL_DEFAULT
from site4drug_inference.common.openrouter_client import (
    DEFAULT_OPENROUTER_APP_TITLE,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL as OPENROUTER_MODEL_FALLBACK,
    ApproxChatRenderer,
    OpenRouterChatClient,
)
from site4drug_inference.common.ptm_backends import PTMBackendError
from site4drug_inference.common.site_output_schema import validate_site_output, with_schema_defaults
from site4drug_inference.common.tinker_sampling import build_sampling_params, sampling_seed_supported
from site4drug_inference.common.token_budget import evaluate_budget
from site4drug_inference.demo.multi_agent_reasoner import run_multi_agent_reasoning
from site4drug_inference.demo.orchestrator import LightweightReActOrchestrator
from site4drug_inference.demo.output_parser import parse_with_single_repair
from site4drug_inference.demo.prompt_templates import (
    SITE_SYSTEM_PROMPT,
    build_full_sequence_user_prompt,
)

DEFAULT_TOP_K = 5
DEFAULT_MODE = "auto"
_DEFAULT_OUTPUT_DIR_ENV = os.environ.get("SITE4DRUG_OUTPUT_DIR", "outputs/predictions").strip()
DEFAULT_OUTPUT_DIR = Path(_DEFAULT_OUTPUT_DIR_ENV)
if not DEFAULT_OUTPUT_DIR.is_absolute():
    DEFAULT_OUTPUT_DIR = REPO_ROOT / DEFAULT_OUTPUT_DIR
DEFAULT_MAX_INPUT_TOKENS = 10000
DEFAULT_CANDIDATE_POOL_SIZE = 120
DEFAULT_CHUNK_SIZE_AA = 1800
DEFAULT_CHUNK_OVERLAP_AA = 250
DEFAULT_CANDIDATE_SOURCE = "llm_propose"
DEFAULT_PTM_SOURCE = "musitedeep"
DEFAULT_MOTIF_SOURCE = "remote"
DEFAULT_FAILURE_POLICY = "raw_llm_only"
DEFAULT_REPORT_VIEW = "compact"
DEFAULT_SELF_CONSISTENCY_K = 1
DEFAULT_LLM_PROVIDER = os.environ.get("SITE4DRUG_LLM_PROVIDER", "tinker").strip().lower()
DEFAULT_OPENROUTER_MODEL = (
    os.environ.get("SITE4DRUG_OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL_FALLBACK)
).strip()
DEFAULT_OPENROUTER_BASE_URL_EFFECTIVE = (
    os.environ.get("SITE4DRUG_OPENROUTER_BASE_URL")
    or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
).strip()
DEFAULT_OPENROUTER_REFERER = (
    os.environ.get("SITE4DRUG_OPENROUTER_REFERER") or os.environ.get("OPENROUTER_HTTP_REFERER", "")
).strip()
DEFAULT_OPENROUTER_TITLE = (
    os.environ.get("SITE4DRUG_OPENROUTER_TITLE")
    or os.environ.get("OPENROUTER_TITLE", DEFAULT_OPENROUTER_APP_TITLE)
).strip()
DEFAULT_MUSITEDEEP_API_BASE_URL = (
    os.environ.get("SITE4DRUG_MUSITEDEEP_API_BASE_URL", MUSITEDEEP_API_BASE_URL_DEFAULT).strip().rstrip("/")
)
DEFAULT_IEDB_TABLE_PATH = REPO_ROOT / "data/tcell_regions_with_seq.parquet"
DEFAULT_IEDB_IOU_THRESHOLD = 0.3
COMPACT_PROPOSER_MAX_TOKENS = 1400
SELF_CONSISTENCY_IOU_THRESHOLD = 0.6
VALID_AA = re.compile(r"^[A-Z]+$")
EPITOPE_ALLOWED_LENGTHS = {12, 15, 18, 20}
POCKET_ALLOWED_LENGTHS = {10, 12, 14, 16}

SYSTEM_PROMPT = SITE_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Site4Drug modality-aware prediction")
    parser.add_argument("--uniprot", default="UNKNOWN", help="UniProt accession/label for logs")
    parser.add_argument("--sequence", default=None, help="Raw antigen sequence")
    parser.add_argument("--sequence-file", type=Path, default=None, help="FASTA/plain-text file path")
    parser.add_argument("--interactive", action="store_true", help="Prompt for sequence interactively")
    parser.add_argument(
        "--llm-provider",
        choices=["tinker", "openrouter"],
        default=DEFAULT_LLM_PROVIDER if DEFAULT_LLM_PROVIDER in {"tinker", "openrouter"} else "tinker",
        help="LLM backend for proposal, repair, and panel calls.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Tinker checkpoint path")
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model name for direct baseline inference (used with --use-base-model).",
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use base model inference directly (ignore --checkpoint).",
    )
    parser.add_argument(
        "--openrouter-model",
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter model id (recommended default: {OPENROUTER_MODEL_FALLBACK}).",
    )
    parser.add_argument(
        "--openrouter-base-url",
        default=DEFAULT_OPENROUTER_BASE_URL_EFFECTIVE,
        help="OpenRouter API base URL.",
    )
    parser.add_argument(
        "--openrouter-referer",
        default=DEFAULT_OPENROUTER_REFERER,
        help="Optional OpenRouter HTTP-Referer attribution header.",
    )
    parser.add_argument(
        "--openrouter-title",
        default=DEFAULT_OPENROUTER_TITLE,
        help="Optional OpenRouter application title attribution header.",
    )
    parser.add_argument(
        "--openrouter-timeout",
        type=float,
        default=120.0,
        help="OpenRouter HTTP request timeout in seconds.",
    )
    parser.add_argument("--mode", choices=["auto", "epitope", "pocket"], default=DEFAULT_MODE)
    parser.add_argument(
        "--candidate-source",
        choices=["llm_propose"],
        default=DEFAULT_CANDIDATE_SOURCE,
        help="Primary candidate source.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top candidates to return")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--self-consistency-k",
        type=int,
        default=DEFAULT_SELF_CONSISTENCY_K,
        help="Repeat LLM proposal sampling K times and vote by overlap; 1 preserves current behavior.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Optional base seed for reproducible sampling when the runtime backend supports it.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--candidate-pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    parser.add_argument("--chunk-size-aa", type=int, default=DEFAULT_CHUNK_SIZE_AA)
    parser.add_argument("--chunk-overlap-aa", type=int, default=DEFAULT_CHUNK_OVERLAP_AA)
    parser.add_argument(
        "--repair-with-base-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the base model for JSON repair attempts when the primary model is a checkpoint. "
            "Disable with --no-repair-with-base-model."
        ),
    )
    parser.add_argument(
        "--panel-with-base-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the base model for multi-agent panel calls when primary inference uses a checkpoint. "
            "Disable with --no-panel-with-base-model."
        ),
    )
    parser.add_argument(
        "--ptm-source",
        choices=["musitedeep", "hybrid", "glyco_only", "multi_rule"],
        default=DEFAULT_PTM_SOURCE,
    )
    parser.add_argument("--ptm-policy", choices=["tiered", "hard", "soft"], default="tiered")
    parser.add_argument(
        "--motif-source",
        choices=["remote"],
        default=DEFAULT_MOTIF_SOURCE,
        help="Motif source is remote ScanProsite via Biopython.",
    )
    parser.add_argument(
        "--musitedeep-api-base-url",
        default=DEFAULT_MUSITEDEEP_API_BASE_URL,
        help="MusiteDeep API base URL (default: https://www.musite.net).",
    )
    parser.add_argument("--musitedeep-model-map", default=None, help="Optional MusiteDeep model-map JSON path.")
    parser.add_argument("--no-motif", action="store_true")
    parser.add_argument("--no-iedb-validation", action="store_true")
    parser.add_argument("--iedb-table", type=Path, default=DEFAULT_IEDB_TABLE_PATH)
    parser.add_argument("--iedb-iou-threshold", type=float, default=DEFAULT_IEDB_IOU_THRESHOLD)
    parser.add_argument("--no-online-lookup", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot artifact generation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()

def read_sequence_file(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Sequence file is empty: {path}")
    if lines[0].startswith(">"):
        header = lines[0][1:].strip() or None
        seq = "".join(line for line in lines[1:] if not line.startswith(">"))
        return seq, header
    return "".join(lines), None


def normalize_sequence(raw_sequence: str) -> str:
    sequence = re.sub(r"\s+", "", raw_sequence).upper().replace("*", "")
    if not sequence:
        raise ValueError("Sequence is empty after normalization.")
    if not VALID_AA.fullmatch(sequence):
        raise ValueError("Sequence must contain only amino-acid letters A-Z.")
    return sequence


def interactive_input() -> tuple[str, str]:
    print("UniProt accession or antigen label (press Enter for UNKNOWN): ", end="", flush=True)
    label = input().strip() or "UNKNOWN"
    print("Paste the antigen sequence, then submit an empty line:")
    chunks: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        chunks.append(line.strip())
    sequence = "".join(chunks)
    if not sequence:
        raise ValueError("No sequence provided in interactive mode.")
    return label, sequence


def _sequence_from_fasta_text(fasta_text: str) -> str:
    lines = [line.strip() for line in fasta_text.splitlines() if line.strip()]
    seq = "".join(line for line in lines if not line.startswith(">"))
    if not seq:
        raise ValueError("FASTA text contains no sequence.")
    return normalize_sequence(seq)


def _fetch_uniprot_fasta(uniprot: str, timeout: int = 20) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def resolve_sequence_from_uniprot(uniprot: str, allow_online_lookup: bool = True) -> tuple[str, str]:
    acc = (uniprot or "").strip()
    if not acc or acc == "UNKNOWN":
        raise ValueError("A valid UniProt accession is required for lookup.")

    local_fasta_candidates = [
        REPO_ROOT / "data/trimmed/pocket4drug_posttrain_outputs/uniprot_cache" / f"{acc}.fasta",
        REPO_ROOT / "data/processed/uniprot_cache" / f"{acc}.fasta",
    ]
    for fasta_path in local_fasta_candidates:
        if fasta_path.exists():
            seq, _ = read_sequence_file(fasta_path)
            return normalize_sequence(seq), f"local_fasta:{fasta_path}"

    parquet_path = REPO_ROOT / "data/tcell_regions_with_seq.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path, columns=["antigen_uniprot_acc", "antigen_seq"])
            matched = df[df["antigen_uniprot_acc"] == acc]["antigen_seq"].dropna()
            if not matched.empty:
                return normalize_sequence(str(matched.iloc[0])), f"parquet:{parquet_path}"
        except Exception:
            pass

    if allow_online_lookup:
        fasta_text = _fetch_uniprot_fasta(acc)
        if fasta_text:
            return _sequence_from_fasta_text(fasta_text), "uniprot_rest"

    raise ValueError(
        f"Could not resolve sequence for UniProt accession '{acc}'. "
        "Provide --sequence/--sequence-file or enable online lookup."
    )


def _sanitize_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return clean or "UNKNOWN"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Moderate"
    return "Low"


def _confidence_score_from_text(label: str) -> float:
    text = str(label or "").strip().lower()
    if text == "high":
        return 0.82
    if text == "moderate":
        return 0.62
    if text == "low":
        return 0.38
    return 0.58


def _build_aggregated_reason(
    *,
    candidate: dict[str, Any],
    flags: list[str] | None = None,
) -> str:
    bits: list[str] = []
    mode = str(candidate.get("mode", "") or "").strip().lower()
    if mode:
        bits.append(f"mode={mode}")

    ptm_overlap = candidate.get("ptm_overlap_by_type", {}) or {}
    if isinstance(ptm_overlap, dict) and ptm_overlap:
        ptm_parts = ",".join(f"{k}:{v}" for k, v in sorted(ptm_overlap.items()) if int(_safe_float(v, 0.0)) > 0)
        if ptm_parts:
            bits.append(f"ptm_overlap={ptm_parts}")

    motif_hits = int(_safe_float(candidate.get("motif_hit_count", 0.0)))
    if motif_hits > 0:
        bits.append(f"motif_hits={motif_hits}")

    if bool(candidate.get("overlaps_tm", False)):
        bits.append("tm_overlap=true")
    if bool(candidate.get("overlaps_ptm_mask", False)):
        bits.append("ptm_mask_overlap=true")

    use_flags = list(flags or candidate.get("flags", []) or candidate.get("risk_flags", []) or [])
    if use_flags:
        bits.append(f"risk_flags={','.join(str(f) for f in use_flags)}")

    return "; ".join(bits) if bits else "consensus selection from repeated LLM proposals"

def _ensure_candidate_reasons(
    rows: list[dict[str, Any]],
    *,
    candidate_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    lookup = candidate_lookup or {}
    out: list[dict[str, Any]] = []
    for row in rows or []:
        item = dict(row)
        reason = str(item.get("reason", "") or "").strip()
        if reason:
            item["reason"] = reason
            out.append(item)
            continue
        cid = str(item.get("candidate_id", "") or "").strip()
        base = lookup.get(cid, {})
        merged = dict(base)
        merged.update(item)
        item["reason"] = _build_aggregated_reason(candidate=merged, flags=item.get("flags", []))
        out.append(item)
    return out


def _normalize_candidate_source(candidate_source: str) -> str:
    source = str(candidate_source or "").strip().lower()
    if source in {"llm_propose_full", "deterministic"}:
        return "llm_propose"
    return "llm_propose"


def _is_llm_candidate_source(candidate_source: str) -> bool:
    return str(candidate_source or "").strip().lower() == "llm_propose"


def _is_compact_llm_candidate_source(candidate_source: str) -> bool:
    return str(candidate_source or "").strip().lower() == "llm_propose"


def _span_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    if overlap_start > overlap_end:
        return 0.0
    overlap = overlap_end - overlap_start + 1
    union = (end_a - start_a + 1) + (end_b - start_b + 1) - overlap
    return overlap / max(union, 1)


def _derived_sampling_seed(base_seed: int | None, *, stage_offset: int = 0, attempt_index: int = 0) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(stage_offset) + max(int(attempt_index), 0)


def _peptide_contains(peptide_a: str, peptide_b: str) -> bool:
    a = str(peptide_a or "").strip().upper()
    b = str(peptide_b or "").strip().upper()
    return bool(a and b and (a in b or b in a))


def _self_consistency_rows_match(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    *,
    iou_threshold: float = SELF_CONSISTENCY_IOU_THRESHOLD,
) -> bool:
    if str(row_a.get("mode", "")).strip().lower() != str(row_b.get("mode", "")).strip().lower():
        return False
    start_a = int(_safe_float(row_a.get("start"), -1))
    end_a = int(_safe_float(row_a.get("end"), -1))
    start_b = int(_safe_float(row_b.get("start"), -1))
    end_b = int(_safe_float(row_b.get("end"), -1))
    if _span_iou(start_a, end_a, start_b, end_b) >= float(iou_threshold):
        return True
    return _peptide_contains(str(row_a.get("peptide", "")), str(row_b.get("peptide", "")))


def _self_consistency_exact_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("mode", "")).strip().lower(),
        int(_safe_float(row.get("start"), -1)),
        int(_safe_float(row.get("end"), -1)),
        str(row.get("peptide", "")).strip().upper(),
    )


def _self_consistency_rank_key(node: dict[str, Any]) -> tuple[Any, ...]:
    row = node.get("row", {}) if isinstance(node, dict) else {}
    return (
        max(int(_safe_float(row.get("rank"), 10**6)), 1),
        -_safe_float(row.get("confidence_score"), 0.0),
        -_safe_float(row.get("heuristic_score"), 0.0),
        int(_safe_float(row.get("start"), -1)),
        int(_safe_float(row.get("end"), -1)),
        str(row.get("peptide", "")).strip().upper(),
    )


def _self_consistency_sort_key(cluster: dict[str, Any]) -> tuple[Any, ...]:
    row = cluster.get("representative_row", {}) or {}
    return (
        -int(cluster.get("vote_count", 0)),
        float(cluster.get("avg_rank", 10**6)),
        -float(cluster.get("avg_confidence_score", 0.0)),
        -float(cluster.get("avg_heuristic_score", 0.0)),
        str(row.get("mode", "")).strip().lower(),
        int(_safe_float(row.get("start"), -1)),
        int(_safe_float(row.get("end"), -1)),
        str(row.get("peptide", "")).strip().upper(),
    )


def _build_self_consistency_consensus(
    attempts: list[dict[str, Any]],
    *,
    requested_k: int,
    top_k: int,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    modality_counts = {"epitope": 0, "pocket": 0, "other": 0}
    modality_conf_sums = {"epitope": 0.0, "pocket": 0.0, "other": 0.0}
    parsed_attempt_count = 0
    for attempt in attempts:
        parsed_obj = attempt.get("parsed_obj")
        if isinstance(parsed_obj, dict):
            mode = str(parsed_obj.get("recommended_modality", "")).strip().lower()
            if mode in modality_counts:
                parsed_attempt_count += 1
                modality_counts[mode] += 1
                modality_conf_sums[mode] += _safe_float(parsed_obj.get("modality_confidence"), 0.0)
        for row in list(attempt.get("ranked_candidates", []) or []):
            nodes.append({"attempt_index": int(attempt.get("attempt_index", 0)), "row": dict(row)})

    def _pick_recommended_mode_from_votes() -> tuple[str, float]:
        ordered_modes = sorted(
            modality_counts.keys(),
            key=lambda mode: (
                -int(modality_counts.get(mode, 0)),
                -(
                    modality_conf_sums.get(mode, 0.0)
                    / max(int(modality_counts.get(mode, 0)), 1)
                ),
                mode,
            ),
        )
        mode = ordered_modes[0]
        confidence = round(
            int(modality_counts.get(mode, 0)) / max(parsed_attempt_count, 1),
            3,
        )
        return mode, confidence

    if not nodes:
        recommended_mode = "epitope"
        recommended_confidence = 0.0
        if parsed_attempt_count > 0:
            recommended_mode, recommended_confidence = _pick_recommended_mode_from_votes()
        return {
            "ranked_candidates": [],
            "recommended_modality": recommended_mode,
            "modality_confidence": recommended_confidence,
            "consensus_status": "no_valid_attempts",
            "selected_attempt_index": None,
            "successful_attempts": parsed_attempt_count,
            "modality_votes": dict(modality_counts),
            "final_candidate_votes": [],
        }

    adjacency = {idx: set() for idx in range(len(nodes))}
    indices_by_mode: dict[str, list[int]] = {}
    for idx, node in enumerate(nodes):
        mode = str((node.get("row", {}) or {}).get("mode", "other")).strip().lower()
        indices_by_mode.setdefault(mode, []).append(idx)
    for mode_indices in indices_by_mode.values():
        for left_pos, left in enumerate(mode_indices):
            for right in mode_indices[left_pos + 1 :]:
                if _self_consistency_rows_match(nodes[left]["row"], nodes[right]["row"]):
                    adjacency[left].add(right)
                    adjacency[right].add(left)

    components: list[list[int]] = []
    seen: set[int] = set()
    for idx in range(len(nodes)):
        if idx in seen:
            continue
        stack = [idx]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacency[current] - seen)
        components.append(sorted(component))

    clusters: list[dict[str, Any]] = []
    for component in components:
        component_nodes = [nodes[idx] for idx in component]
        deduped_by_attempt: dict[int, dict[str, Any]] = {}
        for node in component_nodes:
            attempt_index = int(node.get("attempt_index", 0))
            current = deduped_by_attempt.get(attempt_index)
            if current is None or _self_consistency_rank_key(node) < _self_consistency_rank_key(current):
                deduped_by_attempt[attempt_index] = node
        selected_nodes = list(deduped_by_attempt.values())

        def _representative_sort_key(node: dict[str, Any]) -> tuple[Any, ...]:
            row = node.get("row", {}) or {}
            if len(selected_nodes) <= 1:
                avg_iou = 1.0
            else:
                avg_iou = sum(
                    _span_iou(
                        int(_safe_float(row.get("start"), -1)),
                        int(_safe_float(row.get("end"), -1)),
                        int(_safe_float(other["row"].get("start"), -1)),
                        int(_safe_float(other["row"].get("end"), -1)),
                    )
                    for other in selected_nodes
                    if other is not node
                ) / max(len(selected_nodes) - 1, 1)
            return (
                -avg_iou,
                *_self_consistency_rank_key(node),
            )

        representative_node = min(selected_nodes, key=_representative_sort_key)
        representative_row = deepcopy(representative_node.get("row", {}))
        exact_variants = {_self_consistency_exact_key(node.get("row", {})) for node in component_nodes}
        clusters.append(
            {
                "representative_row": representative_row,
                "representative_attempt_index": int(representative_node.get("attempt_index", 0)),
                "vote_count": len(selected_nodes),
                "supporting_attempts": sorted(int(node.get("attempt_index", 0)) for node in selected_nodes),
                "avg_rank": sum(
                    max(int(_safe_float(node["row"].get("rank"), 10**6)), 1) for node in selected_nodes
                ) / max(len(selected_nodes), 1),
                "avg_confidence_score": sum(
                    _safe_float(node["row"].get("confidence_score"), 0.0) for node in selected_nodes
                ) / max(len(selected_nodes), 1),
                "avg_heuristic_score": sum(
                    _safe_float(node["row"].get("heuristic_score"), 0.0) for node in selected_nodes
                ) / max(len(selected_nodes), 1),
                "selected_nodes": selected_nodes,
                "fuzzy_clustered": len(exact_variants) > 1,
                "union_flags": sorted(
                    {
                        str(flag)
                        for node in selected_nodes
                        for flag in list((node.get("row", {}) or {}).get("flags", []) or [])
                    }
                ),
            }
        )

    clusters.sort(key=_self_consistency_sort_key)
    top_clusters = clusters[: max(int(top_k), 1)]
    consensus_status = "majority" if int(top_clusters[0]["vote_count"]) > max(int(requested_k), 1) // 2 else "tie_broken"

    if parsed_attempt_count > 0:
        recommended_mode, recommended_confidence = _pick_recommended_mode_from_votes()
    else:
        recommended_mode = str(top_clusters[0]["representative_row"].get("mode", "epitope") or "epitope")
        recommended_confidence = round(int(top_clusters[0]["vote_count"]) / max(int(requested_k), 1), 3)

    final_rows: list[dict[str, Any]] = []
    final_votes: list[dict[str, Any]] = []
    for rank, cluster in enumerate(top_clusters, start=1):
        vote_count = int(cluster.get("vote_count", 0))
        representative_row = deepcopy(cluster.get("representative_row", {}))
        support_label = f"{vote_count}/{max(int(requested_k), 1)}"
        merged_flags = set(cluster.get("union_flags", []))
        merged_flags.add(f"self_consistency_votes_{vote_count}_of_{max(int(requested_k), 1)}")
        if vote_count <= max(int(requested_k), 1) // 2:
            merged_flags.add("low_consensus")
        if bool(cluster.get("fuzzy_clustered")):
            merged_flags.add("fuzzy_consensus_clustered")
        avg_confidence = min(max(float(cluster.get("avg_confidence_score", 0.0)), 0.0), 1.0)
        representative_row["rank"] = rank
        representative_row["candidate_id"] = f"L_C{rank:04d}"
        representative_row["confidence_score"] = round(avg_confidence, 3)
        representative_row["confidence"] = _confidence_label(avg_confidence)
        representative_row["confidence_source"] = "llm_self_consistency_vote"
        representative_row["confidence_reason"] = (
            f"Selected from {support_label} attempts after fuzzy overlap clustering."
        )
        representative_row["flags"] = sorted(merged_flags)
        final_rows.append(representative_row)
        final_votes.append(
            {
                "rank": rank,
                "candidate_id": representative_row["candidate_id"],
                "mode": representative_row.get("mode"),
                "start": representative_row.get("start"),
                "end": representative_row.get("end"),
                "peptide": representative_row.get("peptide"),
                "vote_count": vote_count,
                "requested_k": max(int(requested_k), 1),
                "supporting_attempts": list(cluster.get("supporting_attempts", [])),
                "avg_rank": round(float(cluster.get("avg_rank", 0.0)), 3),
                "avg_confidence_score": round(float(cluster.get("avg_confidence_score", 0.0)), 3),
                "avg_heuristic_score": round(float(cluster.get("avg_heuristic_score", 0.0)), 4),
                "fuzzy_clustered": bool(cluster.get("fuzzy_clustered")),
            }
        )

    return {
        "ranked_candidates": final_rows,
        "recommended_modality": recommended_mode,
        "modality_confidence": recommended_confidence,
        "consensus_status": consensus_status,
        "selected_attempt_index": int(top_clusters[0].get("representative_attempt_index", 0)) or None,
        "successful_attempts": parsed_attempt_count,
        "modality_votes": dict(modality_counts),
        "final_candidate_votes": final_votes,
    }


def _load_iedb_reference_spans(uniprot: str, iedb_table_path: Path) -> tuple[list[dict], dict]:
    if not iedb_table_path.exists():
        return [], {"status": "missing_table", "source": str(iedb_table_path)}

    try:
        import pandas as pd

        df = pd.read_parquet(
            iedb_table_path,
            columns=["antigen_uniprot_acc", "start", "end", "final_peptide", "n_assays", "p_pos"],
        )
    except Exception as exc:
        return [], {"status": "table_read_failed", "source": str(iedb_table_path), "error": str(exc)}

    try:
        matched = df[df["antigen_uniprot_acc"] == uniprot].copy()
    except Exception:
        matched = df.iloc[0:0].copy()

    spans: list[dict] = []
    for row in matched.itertuples(index=False):
        try:
            start = int(getattr(row, "start"))
            end = int(getattr(row, "end"))
        except Exception:
            continue
        if start < 1 or end < start:
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "final_peptide": str(getattr(row, "final_peptide", "") or ""),
                "n_assays": int(_safe_float(getattr(row, "n_assays", 0), 0.0)),
                "p_pos": float(_safe_float(getattr(row, "p_pos", 0.0), 0.0)),
            }
        )

    dedup = {}
    for row in spans:
        key = (row["start"], row["end"], row["final_peptide"])
        if key not in dedup:
            dedup[key] = row
    out = list(dedup.values())
    out.sort(key=lambda r: (r["start"], r["end"], r["final_peptide"]))
    return out, {"status": "ok", "source": str(iedb_table_path), "n_reference_spans": len(out)}


def _annotate_iedb_support(
    ranked_candidates: list[dict],
    references: list[dict],
    iou_threshold: float,
) -> dict[str, dict]:
    ann: dict[str, dict] = {}
    for c in ranked_candidates:
        cid = str(c.get("candidate_id"))
        start = int(_safe_float(c.get("start"), -1))
        end = int(_safe_float(c.get("end"), -1))
        pep = str(c.get("peptide", "") or "")
        best_iou = 0.0
        best_ref = None
        substring_match = False
        for ref in references:
            rs = int(ref["start"])
            re_ = int(ref["end"])
            rp = str(ref.get("final_peptide", "") or "")
            iou = _span_iou(start, end, rs, re_)
            if iou > best_iou:
                best_iou = iou
                best_ref = ref
            if pep and rp and (pep in rp or rp in pep):
                substring_match = True
        supported = bool(best_iou >= iou_threshold or substring_match)
        ann[cid] = {
            "supported": supported,
            "best_iou": round(best_iou, 4),
            "substring_match": substring_match,
            "best_reference": best_ref,
        }
    return ann


def _validate_raw_output_consistency(raw_text: str, sequence_length: int) -> dict[str, Any]:
    """Check basic consistency of free-form model text against sequence constraints."""
    issues: list[str] = []
    parsed_rows: list[dict[str, Any]] = []
    if not raw_text:
        return {"issues": ["raw_output_empty"], "parsed_candidate_count": 0}

    length_match = re.search(
        r"target protein is\s+(\d+)\s+amino acids long",
        raw_text,
        flags=re.IGNORECASE,
    )
    if length_match:
        claimed = int(length_match.group(1))
        if claimed != int(sequence_length):
            issues.append(f"claimed_sequence_length_mismatch:{claimed}!={sequence_length}")

    row_pattern = re.compile(
        r"Epitope:\s*([A-Za-z0-9]+)\s*[\r\n]+\s*Position:\s*(\d+)\s*-\s*(\d+)\s*\(length:\s*(\d+)\s*aa\)",
        flags=re.IGNORECASE,
    )
    for m in row_pattern.finditer(raw_text):
        epitope = m.group(1).strip()
        start = int(m.group(2))
        end = int(m.group(3))
        claimed_len = int(m.group(4))
        calc_len = end - start + 1

        row_issues: list[str] = []
        if start < 1 or end > int(sequence_length) or start > end:
            row_issues.append("position_out_of_bounds")
        if calc_len != claimed_len:
            row_issues.append("position_length_mismatch")
        if re.fullmatch(r"C\d{3,}", epitope):
            row_issues.append("epitope_is_candidate_id_placeholder")
        if re.fullmatch(r"[A-Z]+", epitope) and len(epitope) != claimed_len:
            row_issues.append("epitope_length_vs_claimed_length_mismatch")
        if row_issues:
            issues.append(
                f"row_issue:{epitope}:{start}-{end}:"
                + ",".join(row_issues)
            )
        parsed_rows.append(
            {
                "epitope": epitope,
                "start": start,
                "end": end,
                "claimed_length": claimed_len,
                "calculated_length": calc_len,
                "row_issues": row_issues,
            }
        )

    return {
        "issues": issues,
        "parsed_candidate_count": len(parsed_rows),
        "rows": parsed_rows[:20],
    }


def _ptm_evidence_text(seq_summary: dict) -> str:
    ptm_summary = seq_summary.get("ptm_summary", {}) or {}
    counts_by_type = ptm_summary.get("counts_by_type", {}) if isinstance(ptm_summary, dict) else {}
    counts_text = ", ".join(f"{k}:{v}" for k, v in sorted(counts_by_type.items())) or "none"
    sites = seq_summary.get("ptm_sites", []) or []
    site_rows = []
    for site in sites[:20]:
        site_rows.append(
            f"- {site.get('ptm_type')} @ {site.get('position')} "
            f"(mask {site.get('mask_start')}-{site.get('mask_end')}, confidence={site.get('rule_confidence','?')})"
        )
    if len(sites) > 20:
        site_rows.append(f"- ... ({len(sites) - 20} additional PTM sites omitted)")
    rows_text = "\n".join(site_rows) if site_rows else "- none"
    return (
        f"PTM summary: total={ptm_summary.get('total_sites', len(sites))}, by_type={counts_text}\n"
        f"PTM sites (typed):\n{rows_text}"
    )


def _motif_evidence_text(seq_summary: dict) -> str:
    motif_summary = seq_summary.get("motif_summary", {}) or {}
    counts_by_name = motif_summary.get("counts_by_motif", {}) if isinstance(motif_summary, dict) else {}
    counts_text = ", ".join(f"{k}:{v}" for k, v in sorted(counts_by_name.items())) or "none"
    hits = seq_summary.get("motif_hits", []) or []
    hit_rows = []
    for hit in hits[:20]:
        hit_rows.append(
            f"- {hit.get('motif_name')} ({hit.get('pattern_id')}) @ {hit.get('start')}-{hit.get('end')} "
            f"[{hit.get('evidence_source', 'remote_scanprosite_biopython')}]"
        )
    if len(hits) > 20:
        hit_rows.append(f"- ... ({len(hits) - 20} additional motif hits omitted)")
    rows_text = "\n".join(hit_rows) if hit_rows else "- none"
    return (
        f"Motif summary: total={motif_summary.get('total_hits', len(hits))}, by_name={counts_text}\n"
        f"Motif hits:\n{rows_text}"
    )


def _preview_text(text: str, limit: int = 800) -> str:
    val = str(text or "")
    if len(val) <= limit:
        return val
    return val[:limit] + "...<truncated>"


def _write_raw_api_artifacts(run_dir: Path, seq_summary: dict[str, Any]) -> dict[str, Any]:
    api_raw = dict(seq_summary.get("api_raw", {}) or {})
    musite_raw = dict(api_raw.get("musitedeep", {}) or {})
    prosite_raw = dict(api_raw.get("scanprosite", {}) or {})

    musite_calls = list(musite_raw.get("raw_calls", []) or [])
    musite_json_path = run_dir / "musitedeep_raw.json"
    musite_json_payload = {
        "status": str(musite_raw.get("status", "not_requested")),
        "api_base_url": str(musite_raw.get("api_base_url", "")),
        "endpoint_urls": list(musite_raw.get("endpoint_urls", []) or []),
        "errors": str(musite_raw.get("errors", "")),
        "request_count": len(musite_calls),
        "raw_calls": musite_calls,
    }
    musite_json_path.write_text(json.dumps(musite_json_payload, indent=2), encoding="utf-8")

    musite_preview = ""
    if musite_calls:
        musite_preview = _preview_text(str(musite_calls[0].get("response_text", "")))
    if not musite_preview:
        musite_preview = _preview_text(str(musite_json_payload.get("errors", "")))

    scan_xml_path = run_dir / "scanprosite_raw.xml"
    scan_meta_path = run_dir / "scanprosite_raw_meta.json"
    scan_xml = str(prosite_raw.get("raw_xml", "") or "")
    scan_meta = {
        "status": str(prosite_raw.get("status", "not_requested")),
        "n_hits": int(_safe_float(prosite_raw.get("n_hits", 0), 0.0)),
        "warning": str(prosite_raw.get("warning", "") or ""),
        "error": str(prosite_raw.get("error", "") or ""),
    }
    scan_xml_path.write_text(scan_xml, encoding="utf-8")
    scan_meta_path.write_text(json.dumps(scan_meta, indent=2), encoding="utf-8")

    scan_preview = _preview_text(scan_xml)
    if not scan_preview:
        scan_preview = _preview_text(scan_meta.get("error", "") or scan_meta.get("warning", ""))

    return {
        "musitedeep": {
            "status": musite_json_payload["status"],
            "request_count": int(musite_json_payload["request_count"]),
            "artifact_path": musite_json_path.name,
            "preview": musite_preview,
            "errors": musite_json_payload["errors"],
        },
        "scanprosite": {
            "status": scan_meta["status"],
            "n_hits": int(scan_meta["n_hits"]),
            "artifact_path": scan_xml_path.name,
            "preview": scan_preview,
            "warning": scan_meta["warning"],
            "error": scan_meta["error"],
            "meta_artifact_path": scan_meta_path.name,
        },
    }


def _auto_mode_policy(seq_summary: dict) -> dict[str, Any]:
    """Deterministic auto-mode rule for epitope vs pocket.

    Rule requested by project:
    1) Non-membrane protein -> pocket
    2) Membrane protein (channel-like) -> pocket
    3) Membrane protein (general) -> epitope
    """
    tm_regions = seq_summary.get("tm_regions", []) or []
    n_tm = len(tm_regions)
    motif_hits = seq_summary.get("motif_hits", []) or []
    motif_names = " ".join(str(h.get("motif_name", "")).lower() for h in motif_hits if isinstance(h, dict))
    channel_keywords = {
        "channel",
        "pore",
        "porin",
        "aquaporin",
        "ion",
        "transporter",
        "transport",
        "voltage",
    }
    keyword_hit = any(k in motif_names for k in channel_keywords)
    channel_like = n_tm >= 4 or keyword_hit

    if n_tm == 0:
        return {
            "mode": "pocket",
            "confidence": 0.90,
            "reason": "auto-policy: non-membrane protein (0 TM regions) -> pocket",
            "n_tm_regions": n_tm,
            "channel_like": False,
        }
    if channel_like:
        return {
            "mode": "pocket",
            "confidence": 0.85,
            "reason": (
                "auto-policy: membrane channel-like protein "
                f"({n_tm} TM regions, channel_like={channel_like}) -> pocket"
            ),
            "n_tm_regions": n_tm,
            "channel_like": True,
        }
    return {
        "mode": "epitope",
        "confidence": 0.85,
        "reason": f"auto-policy: membrane general protein ({n_tm} TM regions, not channel-like) -> epitope",
        "n_tm_regions": n_tm,
        "channel_like": False,
    }


def _build_full_sequence_messages(
    uniprot: str,
    sequence: str,
    mode: str,
    top_k: int,
    seq_summary: dict,
    candidate_source: str,
    auto_mode_policy: dict[str, Any] | None = None,
) -> list[dict]:
    ptm_text = _ptm_evidence_text(seq_summary)
    motif_text = _motif_evidence_text(seq_summary)
    user = build_full_sequence_user_prompt(
        uniprot=uniprot,
        sequence=sequence,
        mode=mode,
        top_k=top_k,
        seq_summary=seq_summary,
        candidate_source=candidate_source,
        auto_mode_policy=auto_mode_policy,
        ptm_text=ptm_text,
        motif_text=motif_text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _sample_text(
    sampling_client,
    renderer,
    tokenizer,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    sampling_seed: int | None = None,
) -> str:
    if hasattr(sampling_client, "sample_messages"):
        stop = renderer.get_stop_sequences() if hasattr(renderer, "get_stop_sequences") else None
        return sampling_client.sample_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            sampling_seed=sampling_seed,
        )

    from tinker import types

    params, _ = build_sampling_params(
        types,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
        sampling_seed=sampling_seed,
    )
    prompt = renderer.build_generation_prompt(messages)
    result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
    return tokenizer.decode(result.sequences[0].tokens)


def _deterministic_fallback_output(
    candidates: list[dict],
    requested_mode: str,
    top_k: int,
    token_strategy: str,
) -> dict:
    mode = requested_mode if requested_mode in {"epitope", "pocket"} else "epitope"
    top = sorted(candidates, key=lambda c: c.get("heuristic_score", 0.0), reverse=True)[:top_k]
    top_scores = [_safe_float(c.get("heuristic_score"), 0.0) for c in top]
    min_score = min(top_scores) if top_scores else 0.0
    max_score = max(top_scores) if top_scores else 1.0
    denom = max(max_score - min_score, 1e-6)
    ranked = []
    evidence = []
    risk_flags: list[str] = []
    for i, c in enumerate(top, start=1):
        flags = c.get("risk_flags", [])
        risk_flags.extend(flags)
        hscore = _safe_float(c.get("heuristic_score"), 0.0)
        norm = (hscore - min_score) / denom
        conf_score = max(min(0.35 + 0.35 * norm, 0.7), 0.25)
        ranked.append(
            {
                "rank": i,
                "candidate_id": c.get("candidate_id"),
                "start": c.get("start"),
                "end": c.get("end"),
                "peptide": c.get("peptide"),
                "confidence": _confidence_label(conf_score),
                "confidence_score": round(conf_score, 3),
                "confidence_source": "heuristic_fallback",
                "confidence_reason": "LLM JSON unavailable; confidence derived from normalized heuristic score.",
                "flags": flags,
                "reason": "deterministic fallback ranking",
                "mode": c.get("mode", mode),
            }
        )
        evidence.append(
            {
                "candidate_id": c.get("candidate_id"),
                "evidence": [
                    f"mean_hydropathy={c.get('mean_hydropathy'):.2f}",
                    f"overlaps_tm={bool(c.get('overlaps_tm'))}",
                    f"overlaps_ptm_mask={bool(c.get('overlaps_ptm_mask'))}",
                    f"heuristic_score={c.get('heuristic_score'):.4f}",
                ],
            }
        )
    return with_schema_defaults(
        {
            "recommended_modality": mode,
            "modality_confidence": 0.5,
            "ranked_candidates": ranked,
            "candidate_evidence": evidence,
            "risk_flags": sorted(set(risk_flags + ["deterministic_fallback"])),
            "agent_traces": {},
            "feature_provenance": {},
            "token_strategy_used": token_strategy,
            "audit_log": {
                "warnings": ["LLM output could not be parsed; deterministic fallback used."],
                "events": [],
            },
        }
    )


def _merge_chunk_candidates(
    sequence: str,
    mode: str,
    candidate_pool_size: int,
    chunk_size_aa: int,
    chunk_overlap_aa: int,
    ptm_source: str = DEFAULT_PTM_SOURCE,
    ptm_policy: str = "tiered",
    motif_source: str = DEFAULT_MOTIF_SOURCE,
    use_motif: bool = True,
    musitedeep_api_base_url: str = DEFAULT_MUSITEDEEP_API_BASE_URL,
    musitedeep_model_map: str | None = None,
) -> list[dict]:
    chunks = slice_sequence_with_overlap(sequence, chunk_size=chunk_size_aa, overlap=chunk_overlap_aa)
    merged: list[dict] = []
    seen = set()
    chunk_top = max(8, candidate_pool_size // max(len(chunks), 1))
    modes = ["epitope", "pocket"] if mode == "auto" else [mode]
    for chunk_start, _, chunk_seq in chunks:
        for chunk_mode in modes:
            bundle = build_candidate_features(
                chunk_seq,
                mode=chunk_mode,
                top_n=chunk_top,
                stride=4,
                ptm_source=ptm_source,
                ptm_policy=ptm_policy,
                motif_source=motif_source,
                use_motif=use_motif,
                musitedeep_api_base_url=musitedeep_api_base_url,
                musitedeep_model_map=musitedeep_model_map,
                strict_ptm_backend=str(ptm_source).strip().lower() == "musitedeep",
            )
            for cand in bundle["candidates"]:
                global_start = chunk_start + int(cand["start"]) - 1
                global_end = chunk_start + int(cand["end"]) - 1
                key = (chunk_mode, global_start, global_end)
                if key in seen:
                    continue
                seen.add(key)
                out = dict(cand)
                out["start"] = global_start
                out["end"] = global_end
                prefix = "E" if chunk_mode == "epitope" else "P"
                out["candidate_id"] = f"{prefix}_{cand['candidate_id']}_{chunk_start}"
                merged.append(out)
    merged.sort(key=lambda c: c.get("heuristic_score", 0.0), reverse=True)
    return merged[:candidate_pool_size]


def _inject_ranked_candidate_details(parsed: dict, candidate_lookup: dict[str, dict], top_k: int) -> dict:
    ranked = parsed.get("ranked_candidates", [])
    enriched = []
    seen = set()
    candidates = list(candidate_lookup.values())

    def _resolve_candidate(item: dict) -> dict | None:
        cid = item.get("candidate_id")
        if cid in candidate_lookup:
            return candidate_lookup[cid]

        start = int(_safe_float(item.get("start"), -1))
        end = int(_safe_float(item.get("end"), -1))
        peptide = str(item.get("peptide", "")).upper()
        if end < start:
            start, end = -1, -1

        exact_span = []
        if start > 0 and end > 0:
            exact_span = [c for c in candidates if int(c.get("start", -1)) == start and int(c.get("end", -1)) == end]
            if exact_span:
                exact_span.sort(key=lambda c: _safe_float(c.get("heuristic_score"), 0.0), reverse=True)
                return exact_span[0]

        exact_peptide = []
        if peptide:
            exact_peptide = [c for c in candidates if str(c.get("peptide", "")).upper() == peptide]
            if exact_peptide:
                exact_peptide.sort(key=lambda c: _safe_float(c.get("heuristic_score"), 0.0), reverse=True)
                return exact_peptide[0]

        if start <= 0 or end <= 0:
            return None

        # Fuzzy span recovery for legacy text outputs that omit candidate IDs.
        best = None
        best_iou = 0.0
        for c in candidates:
            cs = int(c.get("start", -1))
            ce = int(c.get("end", -1))
            if ce < cs:
                continue
            inter = max(0, min(end, ce) - max(start, cs) + 1)
            if inter <= 0:
                continue
            union = (end - start + 1) + (ce - cs + 1) - inter
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best = c
        if best_iou >= 0.45:
            return best
        return None

    for item in sorted(ranked, key=lambda x: x.get("rank", 9999)):
        base = _resolve_candidate(item)
        if not base:
            continue
        cid = base.get("candidate_id")
        if cid in seen:
            continue
        seen.add(cid)
        raw_conf = item.get("confidence", "Moderate")
        conf_score = min(
            max(_safe_float(item.get("confidence_score"), _confidence_score_from_text(raw_conf)), 0.0),
            1.0,
        )
        enriched.append(
            {
                "rank": len(enriched) + 1,
                "candidate_id": cid,
                "start": int(base["start"]),
                "end": int(base["end"]),
                "peptide": base["peptide"],
                "confidence": _confidence_label(conf_score),
                "confidence_score": round(conf_score, 3),
                "confidence_source": "llm_self_estimate",
                "confidence_reason": item.get("confidence_reason", "Parsed from LLM output."),
                "flags": item.get("flags", base.get("risk_flags", [])),
                "reason": item.get("reason", ""),
                "mode": base.get("mode", "epitope"),
            }
        )
        if len(enriched) >= top_k:
            break
    parsed["ranked_candidates"] = enriched
    return parsed


def _ptm_overlap_by_type_from_summary(start: int, end: int, ptm_sites: list[dict]) -> dict[str, int]:
    """Count PTM typed mask overlaps for an inclusive candidate span."""
    out: dict[str, int] = {}
    for site in ptm_sites:
        mask_start = int(_safe_float(site.get("mask_start"), -1))
        mask_end = int(_safe_float(site.get("mask_end"), -1))
        if mask_start <= 0 or mask_end < mask_start:
            pos = int(_safe_float(site.get("position"), -1))
            mask_start = pos
            mask_end = pos
        if mask_start <= 0 or mask_end < mask_start:
            continue
        if end < mask_start or start > mask_end:
            continue
        ptm_type = str(site.get("ptm_type") or "unknown")
        out[ptm_type] = out.get(ptm_type, 0) + 1
    return out


def _proposal_length_allowed(mode: str, length: int) -> bool:
    if mode == "epitope":
        return length in EPITOPE_ALLOWED_LENGTHS
    if mode == "pocket":
        return length in POCKET_ALLOWED_LENGTHS
    return length in EPITOPE_ALLOWED_LENGTHS.union(POCKET_ALLOWED_LENGTHS)


def _find_all_occurrences(sequence: str, peptide: str) -> list[int]:
    """Return 1-indexed start positions of exact peptide occurrences in sequence."""
    out: list[int] = []
    if not peptide:
        return out
    i = 0
    while True:
        idx = sequence.find(peptide, i)
        if idx < 0:
            break
        out.append(idx + 1)
        i = idx + 1
    return out


def _build_deterministic_reference_ranking(
    *,
    candidates: list[dict],
    top_k: int,
    preferred_mode: str | None = None,
) -> list[dict]:
    ranked: list[dict] = []
    mode = str(preferred_mode or "").strip().lower()
    sorted_candidates = sorted(candidates, key=lambda c: _safe_float(c.get("heuristic_score"), 0.0), reverse=True)
    if mode in {"epitope", "pocket"}:
        sorted_candidates = [
            *[c for c in sorted_candidates if str(c.get("mode", "")).lower() == mode],
            *[c for c in sorted_candidates if str(c.get("mode", "")).lower() != mode],
        ]
    top = sorted_candidates[: max(1, top_k)]
    top_scores = [_safe_float(c.get("heuristic_score"), 0.0) for c in top]
    min_score = min(top_scores) if top_scores else 0.0
    max_score = max(top_scores) if top_scores else 1.0
    denom = max(max_score - min_score, 1e-6)
    for c in top:
        hscore = _safe_float(c.get("heuristic_score"), 0.0)
        norm = (hscore - min_score) / denom
        conf_score = max(min(0.35 + 0.35 * norm, 0.7), 0.25)
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "candidate_id": c.get("candidate_id"),
                "start": c.get("start"),
                "end": c.get("end"),
                "peptide": c.get("peptide"),
                "confidence": _confidence_label(conf_score),
                "confidence_score": round(conf_score, 3),
                "confidence_source": "deterministic_heuristic",
                "confidence_reason": "Deterministic heuristic ranking from constraint feature engine.",
                "flags": c.get("risk_flags", []),
                "reason": "deterministic feature ranking",
                "mode": c.get("mode", "epitope"),
            }
        )
    return ranked


def _enrich_span_candidate(
    *,
    sequence: str,
    start: int,
    end: int,
    mode: str,
    seq_summary: dict,
    ptm_policy: str,
) -> dict:
    peptide = sequence[start - 1 : end]
    scoring_mode = "pocket" if mode == "pocket" else "epitope"
    mean_h = region_hydropathy(sequence, start, end)
    tm_regions = seq_summary.get("tm_regions", []) or []
    ptm_sites = seq_summary.get("ptm_sites", []) or []
    motif_hits = seq_summary.get("motif_hits", []) or []
    cysteine_positions = seq_summary.get("cysteine_positions", []) or []

    overlaps_tm = region_overlaps(start, end, tm_regions)
    ptm_masks = [
        (int(_safe_float(site.get("mask_start"), -1)), int(_safe_float(site.get("mask_end"), -1)))
        for site in ptm_sites
        if int(_safe_float(site.get("mask_start"), -1)) > 0 and int(_safe_float(site.get("mask_end"), -1)) > 0
    ]
    ptm_overlap_by_type = _ptm_overlap_by_type_from_summary(start, end, ptm_sites)
    overlaps_ptm = region_overlaps(start, end, ptm_masks)
    ptm_total = sum(int(v) for v in ptm_overlap_by_type.values())
    span_len = max(end - start + 1, 1)
    ptm_density = ptm_total / span_len
    motif_hits_overlapping = _motif_hits_in_region(start, end, motif_hits)
    motif_hit_count = len(motif_hits_overlapping)
    cys_count = count_in_region(start, end, cysteine_positions)
    comp = aa_composition(peptide)
    risk_flags = _risk_flags_for_region(
        mean_h=mean_h,
        overlaps_tm=overlaps_tm,
        ptm_overlap_by_type=ptm_overlap_by_type,
        ptm_density=ptm_density,
        cysteine_count=cys_count,
        motif_hits_overlapping=motif_hits_overlapping,
    )
    ptm_penalty = _ptm_penalty(
        mode=scoring_mode,
        ptm_overlap_by_type=ptm_overlap_by_type,
        ptm_density=ptm_density,
        ptm_policy=ptm_policy,
    )
    heuristic_score = _heuristic_score(
        mode=scoring_mode,
        mean_h=mean_h,
        overlaps_tm=overlaps_tm,
        comp=comp,
        ptm_penalty=ptm_penalty,
        motif_hits_overlapping=motif_hits_overlapping,
    )
    return {
        "start": start,
        "end": end,
        "peptide": peptide,
        "mode": mode,
        "mean_hydropathy": round(_safe_float(mean_h), 4),
        "hydrophobic_fraction": round(_safe_float(comp.get("hydrophobic_fraction")), 4),
        "polar_fraction": round(_safe_float(comp.get("polar_fraction")), 4),
        "positive_fraction": round(_safe_float(comp.get("positive_fraction")), 4),
        "negative_fraction": round(_safe_float(comp.get("negative_fraction")), 4),
        "cysteine_count": int(cys_count),
        "overlaps_tm": bool(overlaps_tm),
        "overlaps_ptm_mask": bool(overlaps_ptm),
        "ptm_overlap_by_type": ptm_overlap_by_type,
        "ptm_density": round(_safe_float(ptm_density), 4),
        "motif_hits_overlapping": motif_hits_overlapping,
        "motif_hit_count": int(motif_hit_count),
        "risk_flags": risk_flags,
        "heuristic_score": round(_safe_float(heuristic_score), 4),
    }

def _infer_mode_from_constraint_score(
    *,
    sequence: str,
    start: int,
    end: int,
    seq_summary: dict,
    ptm_policy: str,
) -> str:
    """Infer epitope vs pocket by comparing deterministic span-level heuristic scores."""
    epi = _enrich_span_candidate(
        sequence=sequence,
        start=start,
        end=end,
        mode="epitope",
        seq_summary=seq_summary,
        ptm_policy=ptm_policy,
    )
    poc = _enrich_span_candidate(
        sequence=sequence,
        start=start,
        end=end,
        mode="pocket",
        seq_summary=seq_summary,
        ptm_policy=ptm_policy,
    )
    return "pocket" if _safe_float(poc.get("heuristic_score"), 0.0) > _safe_float(epi.get("heuristic_score"), 0.0) else "epitope"


def _resolve_proposal_mode(
    *,
    raw_mode: str,
    requested_mode: str,
    parsed_recommended_mode: str,
    span_len: int,
    sequence: str,
    start: int,
    end: int,
    seq_summary: dict,
    ptm_policy: str,
) -> tuple[str, str | None]:
    """Resolve candidate mode for llm_propose rows, repairing legacy/malformed rows when possible."""
    mode = str(raw_mode or "").strip().lower()
    if mode in {"epitope", "pocket", "other"}:
        return mode, None

    requested = str(requested_mode or "").strip().lower()
    if requested in {"epitope", "pocket"}:
        return requested, "mode_repaired_from_requested_mode"

    rec = str(parsed_recommended_mode or "").strip().lower()
    if rec in {"epitope", "pocket"}:
        return rec, "mode_repaired_from_recommended_modality"

    epi_len = span_len in EPITOPE_ALLOWED_LENGTHS
    poc_len = span_len in POCKET_ALLOWED_LENGTHS
    if epi_len and not poc_len:
        return "epitope", "mode_repaired_from_length_policy"
    if poc_len and not epi_len:
        return "pocket", "mode_repaired_from_length_policy"

    inferred = _infer_mode_from_constraint_score(
        sequence=sequence,
        start=start,
        end=end,
        seq_summary=seq_summary,
        ptm_policy=ptm_policy,
    )
    return inferred, "mode_inferred_from_constraint_score"


def _validate_and_enrich_llm_proposals(
    *,
    parsed_obj: dict,
    sequence: str,
    requested_mode: str,
    top_k: int,
    seq_summary: dict,
    deterministic_candidates: list[dict],
    ptm_policy: str,
    allow_fill: bool = False,
) -> tuple[dict, dict[str, Any]]:
    _ = deterministic_candidates
    _ = allow_fill
    raw_ranked = list(parsed_obj.get("ranked_candidates", []) or [])
    total_proposed = len(raw_ranked)
    errors: list[str] = []
    seen: set[tuple[str, int, int, str]] = set()
    valid_rows: list[dict] = []

    parsed_recommended_mode = str(parsed_obj.get("recommended_modality", "")).strip().lower()
    for idx, row in enumerate(sorted(raw_ranked, key=lambda x: x.get("rank", 9999)), start=1):
        raw_mode = str(row.get("mode", "")).strip().lower()
        start = int(_safe_float(row.get("start"), -1))
        end = int(_safe_float(row.get("end"), -1))
        peptide = str(row.get("peptide", "") or "").strip().upper()
        if not peptide:
            errors.append(f"proposal_{idx}:missing_peptide")
            continue
        span_valid = 1 <= start <= end <= len(sequence)
        repaired_from_peptide_search = False
        repaired_from_span_slice = False
        if not span_valid:
            occurrences = _find_all_occurrences(sequence, peptide)
            if not occurrences:
                errors.append(f"proposal_{idx}:span_out_of_bounds")
                continue
            start = min(occurrences, key=lambda s: abs(s - start)) if start > 0 else occurrences[0]
            end = start + len(peptide) - 1
            if not (1 <= start <= end <= len(sequence)):
                errors.append(f"proposal_{idx}:span_repair_failed")
                continue
            repaired_from_peptide_search = True

        expected_peptide = sequence[start - 1 : end]
        if peptide != expected_peptide:
            occurrences = _find_all_occurrences(sequence, peptide)
            if occurrences:
                start = min(occurrences, key=lambda s: abs(s - start))
                end = start + len(peptide) - 1
                if not (1 <= start <= end <= len(sequence)):
                    errors.append(f"proposal_{idx}:span_repair_failed")
                    continue
                repaired_from_peptide_search = True
            else:
                peptide = expected_peptide
                repaired_from_span_slice = True
                errors.append(f"proposal_{idx}:peptide_repaired_from_span")

        span_len = end - start + 1
        mode, mode_repair_reason = _resolve_proposal_mode(
            raw_mode=raw_mode,
            requested_mode=requested_mode,
            parsed_recommended_mode=parsed_recommended_mode,
            span_len=span_len,
            sequence=sequence,
            start=start,
            end=end,
            seq_summary=seq_summary,
            ptm_policy=ptm_policy,
        )
        if mode not in {"epitope", "pocket", "other"}:
            errors.append(f"proposal_{idx}:invalid_mode")
            continue
        if mode_repair_reason:
            errors.append(f"proposal_{idx}:{mode_repair_reason}")
        key = (mode, start, end, peptide)
        if key in seen:
            errors.append(f"proposal_{idx}:duplicate")
            continue
        seen.add(key)

        enriched = _enrich_span_candidate(
            sequence=sequence,
            start=start,
            end=end,
            mode=mode,
            seq_summary=seq_summary,
            ptm_policy=ptm_policy,
        )
        raw_conf = row.get("confidence", "Moderate")
        conf_score = min(
            max(_safe_float(row.get("confidence_score"), _confidence_score_from_text(str(raw_conf))), 0.0),
            1.0,
        )
        valid_rows.append(
            {
                "rank": len(valid_rows) + 1,
                "candidate_id": f"L_C{len(valid_rows) + 1:04d}",
                "start": enriched["start"],
                "end": enriched["end"],
                "peptide": enriched["peptide"],
                "confidence": _confidence_label(conf_score),
                "confidence_score": round(conf_score, 3),
                "confidence_source": "llm_self_estimate",
                "confidence_reason": row.get("confidence_reason", "LLM-proposed candidate accepted by validators."),
                "flags": list(row.get("flags", [])) or list(enriched["risk_flags"]),
                "reason": row.get("reason", ""),
                "mode": mode,
                "mean_hydropathy": enriched["mean_hydropathy"],
                "hydrophobic_fraction": enriched["hydrophobic_fraction"],
                "polar_fraction": enriched["polar_fraction"],
                "positive_fraction": enriched["positive_fraction"],
                "negative_fraction": enriched["negative_fraction"],
                "cysteine_count": enriched["cysteine_count"],
                "overlaps_tm": enriched["overlaps_tm"],
                "overlaps_ptm_mask": enriched["overlaps_ptm_mask"],
                "ptm_overlap_by_type": enriched["ptm_overlap_by_type"],
                "ptm_density": enriched["ptm_density"],
                "motif_hits_overlapping": enriched["motif_hits_overlapping"],
                "motif_hit_count": enriched["motif_hit_count"],
                "heuristic_score": enriched["heuristic_score"],
            }
        )
        if repaired_from_peptide_search:
            valid_rows[-1]["flags"] = sorted(set([*valid_rows[-1].get("flags", []), "span-repaired-from-peptide"]))
            errors.append(f"proposal_{idx}:span_repaired_from_peptide_search")
        if repaired_from_span_slice:
            valid_rows[-1]["flags"] = sorted(set([*valid_rows[-1].get("flags", []), "peptide-repaired-from-span"]))
        if mode_repair_reason:
            valid_rows[-1]["flags"] = sorted(set([*valid_rows[-1].get("flags", []), "mode-repaired"]))
        if not _proposal_length_allowed(mode, span_len):
            valid_rows[-1]["flags"] = sorted(set([*valid_rows[-1].get("flags", []), f"noncanonical-length-{span_len}"]))
            errors.append(f"proposal_{idx}:noncanonical_length:{span_len}")

    llm_valid_rows = [dict(r) for r in valid_rows]
    if len(valid_rows) > top_k:
        extra = len(valid_rows) - top_k
        errors.extend([f"proposal_overflow_trimmed:{extra}"])
        valid_rows = valid_rows[:top_k]
        llm_valid_rows = llm_valid_rows[:top_k]

    rec_mode = str(parsed_obj.get("recommended_modality", "")).strip().lower()
    if requested_mode in {"epitope", "pocket"}:
        rec_mode = requested_mode
    if rec_mode not in {"epitope", "pocket", "other"}:
        rec_mode = "epitope"

    dropped = max(total_proposed - len(valid_rows), 0)
    ranked: list[dict] = []
    risk_flags: set[str] = set()
    for row in valid_rows[:top_k]:
        row["rank"] = len(ranked) + 1
        for flag in row.get("flags", []):
            risk_flags.add(str(flag))
        ranked.append(dict(row))

    parsed_obj["recommended_modality"] = rec_mode
    parsed_obj["ranked_candidates"] = ranked
    parsed_obj["risk_flags"] = sorted(risk_flags.union(set(parsed_obj.get("risk_flags", []))))
    return parsed_obj, {
        "llm_proposal_total": int(total_proposed),
        "llm_proposal_valid": int(len(ranked)),
        "llm_proposal_dropped": int(dropped),
        "llm_proposal_fill_count": 0,
        "proposal_validation_errors": errors,
        "llm_valid_rows": llm_valid_rows,
    }

def _materialize_panel_ranking(
    ranking: list[dict],
    candidate_lookup: dict[str, dict],
    *,
    top_k: int,
    default_modality_confidence: float,
    decision_fallback: bool,
    fallback_mode: str,
) -> list[dict]:
    """Convert decision-agent ranking rows into full candidate records."""
    ranked: list[dict] = []
    seen: set[str] = set()
    for row in ranking:
        cid = row.get("candidate_id")
        if not cid or cid in seen or cid not in candidate_lookup:
            continue
        cand = candidate_lookup[cid]
        conf_score = min(
            max(
                _safe_float(
                    row.get("confidence_score"),
                    default_modality_confidence,
                ),
                0.0,
            ),
            1.0,
        )
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "candidate_id": cid,
                "start": cand["start"],
                "end": cand["end"],
                "peptide": cand["peptide"],
                "confidence": _confidence_label(conf_score),
                "confidence_score": round(conf_score, 3),
                "confidence_source": (
                    "fallback_consensus"
                    if decision_fallback
                    else str(cand.get("confidence_source", "llm_self_estimate") or "llm_self_estimate")
                ),
                "confidence_reason": row.get(
                    "confidence_reason",
                    "Decision-agent self-estimate from specialist evidence.",
                ),
                "flags": cand.get("risk_flags", []),
                "reason": row.get("reason", ""),
                "mode": cand.get("mode", fallback_mode),
            }
        )
        seen.add(cid)
        if len(ranked) >= top_k:
            break
    return ranked


def _select_panel_candidates(
    *,
    candidate_source: str,
    llm_generated_candidates: list[dict],
    deterministic_candidates: list[dict],
) -> tuple[list[dict], str]:
    """Build panel input pool from enriched LLM-generated candidates."""
    _ = candidate_source
    _ = deterministic_candidates

    normalized: list[dict] = []
    seen_ids: set[str] = set()
    for row in llm_generated_candidates or []:
        cid = str(row.get("candidate_id", "")).strip()
        if not cid or cid in seen_ids:
            continue
        start = int(_safe_float(row.get("start"), -1))
        end = int(_safe_float(row.get("end"), -1))
        peptide = str(row.get("peptide", "") or "")
        if not (1 <= start <= end) or not peptide:
            continue
        mode = str(row.get("mode", "other")).strip().lower()
        if mode not in {"epitope", "pocket", "other"}:
            mode = "other"
        risk_flags = row.get("risk_flags", [])
        if not isinstance(risk_flags, list) or not risk_flags:
            risk_flags = row.get("flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = []
        motif_hits = row.get("motif_hits_overlapping", [])
        if not isinstance(motif_hits, list):
            motif_hits = []
        ptm_overlap = row.get("ptm_overlap_by_type", {})
        if not isinstance(ptm_overlap, dict):
            ptm_overlap = {}
        normalized.append(
            {
                "candidate_id": cid,
                "mode": mode,
                "start": start,
                "end": end,
                "peptide": peptide,
                "mean_hydropathy": _safe_float(row.get("mean_hydropathy"), 0.0),
                "hydrophobic_fraction": _safe_float(row.get("hydrophobic_fraction"), 0.0),
                "polar_fraction": _safe_float(row.get("polar_fraction"), 0.0),
                "positive_fraction": _safe_float(row.get("positive_fraction"), 0.0),
                "negative_fraction": _safe_float(row.get("negative_fraction"), 0.0),
                "cysteine_count": int(_safe_float(row.get("cysteine_count"), 0.0)),
                "overlaps_tm": bool(row.get("overlaps_tm", False)),
                "overlaps_ptm_mask": bool(row.get("overlaps_ptm_mask", False)),
                "ptm_overlap_by_type": ptm_overlap,
                "ptm_density": _safe_float(row.get("ptm_density"), 0.0),
                "motif_hits_overlapping": motif_hits,
                "motif_hit_count": int(_safe_float(row.get("motif_hit_count"), 0.0)),
                "risk_flags": [str(f) for f in risk_flags],
                "heuristic_score": _safe_float(
                    row.get("heuristic_score"),
                    _safe_float(row.get("confidence_score"), 0.0),
                ),
                "confidence_source": str(row.get("confidence_source", "llm_self_estimate") or "llm_self_estimate"),
                "confidence_reason": str(row.get("confidence_reason", "") or ""),
            }
        )
        seen_ids.add(cid)

    if normalized:
        return normalized, "llm_generated"
    return [], "llm_generated_empty"


def _compact_agent_conclusions(run: dict) -> list[tuple[str, str]]:
    traces = run.get("agent_traces", {}) or {}
    out: list[tuple[str, str]] = []
    for key, label in (
        ("bio_agent", "BioAgent"),
        ("chem_agent", "ChemAgent"),
        ("risk_agent", "RiskAgent"),
        ("decision_agent", "DecisionAgent"),
    ):
        trace = traces.get(key, {}) if isinstance(traces, dict) else {}
        parsed = trace.get("parsed", {}) if isinstance(trace, dict) else {}
        summary = str(parsed.get("summary") or parsed.get("rationale") or "").strip()
        if not summary and key == "decision_agent":
            summary = str((run.get("panel_comparison", {}) or {}).get("panel_status", "not_run"))
        out.append((label, summary or "No conclusion available."))
    return out


def _render_markdown_report_compact(run: dict) -> str:
    lines = [
        "# Site4Drug Prediction Report",
        "",
        f"- Run ID: `{run.get('run_id')}`",
        f"- Run status: `{run.get('run_status', 'unknown')}`",
        f"- UniProt/Label: `{run.get('input', {}).get('uniprot')}`",
        f"- Requested mode: `{run.get('input', {}).get('mode_request')}`",
        f"- Recommended modality: `{run.get('recommended_modality')}` ({_safe_float(run.get('modality_confidence'), 0.0):.2f})",
        f"- Candidate source: `{run.get('generation', {}).get('candidate_source')}`",
        f"- PTM backend effective: `{run.get('generation', {}).get('ptm_backend_effective', 'unknown')}`",
        f"- Motif source effective: `{run.get('generation', {}).get('motif_source_effective', 'unknown')}`",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | ID | Mode | Peptide | Position | Confidence | Score | Source | Flags | Reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ranked = list(run.get("ranked_candidates", []) or [])
    if ranked:
        for c in ranked:
            flags = ", ".join(c.get("flags", [])) if c.get("flags") else "-"
            lines.append(
                f"| {c.get('rank')} | `{c.get('candidate_id')}` | `{c.get('mode', '-')}` | "
                f"`{c.get('peptide')}` | {c.get('start')}-{c.get('end')} | {c.get('confidence', '-')} | "
                f"{_safe_float(c.get('confidence_score'), 0.0):.2f} | "
                f"`{c.get('confidence_source', '-')}` | {flags} | {str(c.get('reason', '') or '-')} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
    lines.extend(["", "## Agent Conclusions", ""])
    for label, summary in _compact_agent_conclusions(run):
        lines.append(f"- **{label}**: {summary}")
    lines.extend(
        [
            "",
            "## PTM + Motif Summary",
            "",
            f"- PTM summary: `{run.get('ptm_summary', {})}`",
            f"- Motif summary: `{run.get('motif_summary', {})}`",
        ]
    )
    raw_api = run.get("raw_api_calls", {}) or {}
    mus = raw_api.get("musitedeep", {}) if isinstance(raw_api, dict) else {}
    pro = raw_api.get("scanprosite", {}) if isinstance(raw_api, dict) else {}
    lines.extend(
        [
            "",
            "## Raw API Outputs",
            "",
            f"- MusiteDeep: status=`{mus.get('status', 'unknown')}`, requests=`{mus.get('request_count', 0)}`, artifact=`{mus.get('artifact_path', '-')}`",
            f"- MusiteDeep preview: `{_preview_text(str(mus.get('preview', '') or ''), 220)}`",
            f"- ScanProsite: status=`{pro.get('status', 'unknown')}`, n_hits=`{pro.get('n_hits', 0)}`, artifact=`{pro.get('artifact_path', '-')}`",
            f"- ScanProsite preview: `{_preview_text(str(pro.get('preview', '') or ''), 220)}`",
        ]
    )
    plot_name = str((run.get("plot_artifacts", {}) or {}).get("plot_png_name", "")).strip()
    if plot_name:
        lines.extend(
            [
                "",
                "## Hydropathy + PTM + Candidate Tracks",
                "",
                f"![Hydropathy/PTM/Candidates]({plot_name})",
                "",
                "_Legend: candidate bars use emerald green=epitope, steel blue=pocket, slate gray=other; gold annotation marks PTM risk labels (PTM overlap/PTM dense); PTM dot markers encode confidence (o=high, ^=medium, x=low)._",
            ]
        )
    if run.get("run_status") != "ok" or not ranked:
        lines.extend(
            [
                "",
                "## Raw Model Output",
                "",
                "```text",
                str(run.get("raw_model_output", "")),
                "```",
            ]
        )
    return "\n".join(lines)


def _render_html_report_compact(run: dict) -> str:
    ranked = list(run.get("ranked_candidates", []) or [])
    rows = []
    for c in ranked:
        flags = ", ".join(c.get("flags", [])) if c.get("flags") else "-"
        rows.append(
            "<tr>"
            f"<td>{c.get('rank')}</td>"
            f"<td><code>{html.escape(str(c.get('candidate_id')))}</code></td>"
            f"<td>{html.escape(str(c.get('mode', '-')))}</td>"
            f"<td><code>{html.escape(str(c.get('peptide')))}</code></td>"
            f"<td>{c.get('start')}-{c.get('end')}</td>"
            f"<td>{html.escape(str(c.get('confidence', '-')))}</td>"
            f"<td>{_safe_float(c.get('confidence_score'), 0.0):.2f}</td>"
            f"<td><code>{html.escape(str(c.get('confidence_source', '-')))}</code></td>"
            f"<td>{html.escape(flags)}</td>"
            f"<td>{html.escape(str(c.get('reason', '') or '-'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='10'>No ranked candidates returned.</td></tr>")
    conclusion_rows = []
    for label, summary in _compact_agent_conclusions(run):
        conclusion_rows.append(f"<li><b>{html.escape(label)}</b>: {html.escape(summary)}</li>")
    plot_name = str((run.get("plot_artifacts", {}) or {}).get("plot_png_name", "")).strip()
    plot_html = (
        "<div class='plot-frame'>"
        f"<img src='{html.escape(plot_name)}' alt='Hydropathy PTM candidate tracks plot'/>"
        "</div>"
        if plot_name
        else "<p>No plot generated.</p>"
    )
    raw_api = run.get("raw_api_calls", {}) or {}
    mus = raw_api.get("musitedeep", {}) if isinstance(raw_api, dict) else {}
    pro = raw_api.get("scanprosite", {}) if isinstance(raw_api, dict) else {}
    raw_block = ""
    if run.get("run_status") != "ok" or not ranked:
        raw_block = (
            "<section class='panel'>"
            "<h2>Raw Model Output</h2>"
            f"<pre>{html.escape(str(run.get('raw_model_output', '')))}</pre>"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Site4Drug Prediction Report</title>
  <style>
    :root {{
      --paper: #ffffff;
      --paper-soft: #f7fbff;
      --ink: #17293b;
      --ink-soft: #37526d;
      --line: #c9d7e6;
      --line-strong: #adc2d8;
      --accent: #245a88;
      --accent-soft: #eaf3fc;
      --ok: #1f6f37;
      --shadow: 0 4px 12px rgba(20, 57, 92, 0.08);
      --radius-lg: 12px;
      --radius-md: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Avenir", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.5;
      min-width: 1240px;
      padding: 1.6rem 0 2.4rem;
    }}
    .report {{
      width: 1200px;
      margin: 0 auto;
      display: grid;
      gap: 0.9rem;
    }}
    .hero, .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 1.1rem 1.3rem 1.2rem;
      border-top: 3px solid var(--accent);
    }}
    .kicker {{
      margin: 0;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--accent);
      font-size: 0.72rem;
      font-weight: 700;
    }}
    h1 {{
      margin: 0.35rem 0 0.2rem;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      font-size: 2.4rem;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      max-width: 78ch;
      color: var(--ink-soft);
    }}
    .meta-grid {{
      margin-top: 1rem;
      display: grid;
      gap: 0.6rem;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .meta-item {{
      background: var(--paper-soft);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 0.55rem 0.65rem;
    }}
    .meta-label {{
      display: block;
      font-size: 0.73rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #406386;
      margin-bottom: 0.15rem;
      font-weight: 700;
    }}
    .meta-value {{
      font-size: 0.92rem;
      font-weight: 600;
      color: #16324c;
      word-break: break-word;
    }}
    .panel {{
      padding: 1.15rem 1.2rem;
    }}
    .panel h2 {{
      margin: 0 0 0.55rem;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      font-size: 1.55rem;
      line-height: 1.2;
      color: var(--accent);
    }}
    .panel p {{
      margin: 0.25rem 0 0.65rem;
      color: var(--ink-soft);
    }}
    .table-wrap {{
      width: 100%;
      overflow: hidden;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: var(--paper);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0;
      font-size: 0.92rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 0.6rem 0.65rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--accent-soft);
      color: #154166;
      font-weight: 700;
      white-space: nowrap;
    }}
    tr:hover td {{
      background: #f4f9ff;
    }}
    code {{
      font-family: "SF Mono", "Menlo", "Consolas", monospace;
      font-size: 0.88em;
      background: transparent;
      border: 0;
      border-radius: 0;
      padding: 0;
      color: #204a72;
    }}
    ul {{
      margin: 0;
      padding-left: 1.1rem;
    }}
    pre {{
      margin: 0.5rem 0 0;
      background: #f4f9ff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.7rem;
      overflow-x: auto;
    }}
    .plot-frame {{
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      background: var(--paper);
    }}
    .plot-frame img {{
      width: 100%;
      display: block;
    }}
    .legend {{
      margin-top: 0.5rem;
      color: #355473;
      font-size: 0.9rem;
      line-height: 1.55;
      text-wrap: pretty;
    }}
    .legend .nobr {{
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <main class="report">
    <header class="hero">
      <p class="kicker">Site4Drug</p>
      <h1>Prediction Report</h1>
      <div class="meta-grid">
        <div class="meta-item"><span class="meta-label">Run ID</span><span class="meta-value"><code>{html.escape(str(run.get('run_id')))}</code></span></div>
        <div class="meta-item"><span class="meta-label">Run Status</span><span class="meta-value"><code>{html.escape(str(run.get('run_status', 'unknown')))}</code></span></div>
        <div class="meta-item"><span class="meta-label">UniProt / Label</span><span class="meta-value"><code>{html.escape(str((run.get('input', {}) or {}).get('uniprot')))}</code></span></div>
        <div class="meta-item"><span class="meta-label">Requested Mode</span><span class="meta-value"><code>{html.escape(str((run.get('input', {}) or {}).get('mode_request')))}</code></span></div>
        <div class="meta-item"><span class="meta-label">Recommended Modality</span><span class="meta-value"><code>{html.escape(str(run.get('recommended_modality')))}</code> ({_safe_float(run.get('modality_confidence'), 0.0):.2f})</span></div>
        <div class="meta-item"><span class="meta-label">Candidate Source</span><span class="meta-value"><code>{html.escape(str((run.get('generation', {}) or {}).get('candidate_source')))}</code></span></div>
        <div class="meta-item"><span class="meta-label">PTM Backend</span><span class="meta-value"><code>{html.escape(str((run.get('generation', {}) or {}).get('ptm_backend_effective', 'unknown')))}</code></span></div>
        <div class="meta-item"><span class="meta-label">Motif Source</span><span class="meta-value"><code>{html.escape(str((run.get('generation', {}) or {}).get('motif_source_effective', 'unknown')))}</code></span></div>
      </div>
    </header>

    <section class="panel">
      <h2>Ranked Candidates</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Rank</th><th>ID</th><th>Mode</th><th>Peptide</th><th>Position</th><th>Confidence</th><th>Score</th><th>Source</th><th>Flags</th><th>Reason</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Agent Conclusions</h2>
      <ul>{"".join(conclusion_rows) if conclusion_rows else "<li>No agent conclusions available.</li>"}</ul>
    </section>

    <section class="panel">
      <h2>PTM + Motif Summary</h2>
      <p><b>PTM summary:</b> <code>{html.escape(str(run.get('ptm_summary', {})))}</code></p>
      <p><b>Motif summary:</b> <code>{html.escape(str(run.get('motif_summary', {})))}</code></p>
    </section>

    <section class="panel">
      <h2>Raw API Outputs</h2>
      <p><b>MusiteDeep:</b> status=<code>{html.escape(str(mus.get('status', 'unknown')))}</code>,
         requests=<code>{int(_safe_float(mus.get('request_count', 0), 0.0))}</code>,
         artifact=<code>{html.escape(str(mus.get('artifact_path', '-')))}</code></p>
      <p><b>MusiteDeep preview:</b> <code>{html.escape(_preview_text(str(mus.get('preview', '') or ''), 220))}</code></p>
      <p><b>ScanProsite:</b> status=<code>{html.escape(str(pro.get('status', 'unknown')))}</code>,
         n_hits=<code>{int(_safe_float(pro.get('n_hits', 0), 0.0))}</code>,
         artifact=<code>{html.escape(str(pro.get('artifact_path', '-')))}</code></p>
      <p><b>ScanProsite preview:</b> <code>{html.escape(_preview_text(str(pro.get('preview', '') or ''), 220))}</code></p>
    </section>

    <section class="panel">
      <h2>Hydropathy + PTM + Candidate Tracks</h2>
      {plot_html}
      <p class="legend">Legend: candidate bars use <span class="nobr"><b>emerald green=epitope</b></span>, <span class="nobr"><b>steel blue=pocket</b></span>, <span class="nobr"><b>slate gray=other</b></span>; gold annotation marks PTM risk labels (<span class="nobr"><b>PTM overlap</b></span> or <span class="nobr"><b>PTM dense</b></span>); PTM dot markers encode confidence (<span class="nobr"><b>o=high</b></span>, <span class="nobr"><b>^=medium</b></span>, <span class="nobr"><b>x=low</b></span>).</p>
    </section>
    {raw_block}
  </main>
</body>
</html>
"""


def _render_markdown_report(run: dict, *, report_view: str = DEFAULT_REPORT_VIEW) -> str:
    _ = report_view
    return _render_markdown_report_compact(run)


def _render_html_report(run: dict, *, report_view: str = DEFAULT_REPORT_VIEW) -> str:
    _ = report_view
    return _render_html_report_compact(run)


def run_prediction(
    uniprot: str,
    raw_sequence: str,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    checkpoint: str | None = DEFAULT_CHECKPOINT,
    base_model: str = BASE_MODEL,
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL_EFFECTIVE,
    openrouter_referer: str = DEFAULT_OPENROUTER_REFERER,
    openrouter_title: str = DEFAULT_OPENROUTER_TITLE,
    openrouter_timeout: float = 120.0,
    mode: str = DEFAULT_MODE,
    candidate_source: str = DEFAULT_CANDIDATE_SOURCE,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    self_consistency_k: int = DEFAULT_SELF_CONSISTENCY_K,
    sampling_seed: int | None = None,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    chunk_size_aa: int = DEFAULT_CHUNK_SIZE_AA,
    chunk_overlap_aa: int = DEFAULT_CHUNK_OVERLAP_AA,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    require_api_key: bool = True,
    input_source: str = "provided_sequence",
    enable_plot: bool = True,
    use_multi_agent: bool = True,
    orchestrator_mode: str = "react",
    react_max_steps: int = 8,
    react_max_retries: int = 2,
    repair_with_base_model: bool = True,
    panel_with_base_model: bool = True,
    ptm_source: str = DEFAULT_PTM_SOURCE,
    ptm_policy: str = "tiered",
    motif_source: str = DEFAULT_MOTIF_SOURCE,
    use_motif: bool = True,
    failure_policy: str = DEFAULT_FAILURE_POLICY,
    report_view: str = DEFAULT_REPORT_VIEW,
    musitedeep_api_base_url: str = DEFAULT_MUSITEDEEP_API_BASE_URL,
    musitedeep_model_map: str | None = None,
    use_iedb_validation: bool = True,
    iedb_table_path: Path | str = DEFAULT_IEDB_TABLE_PATH,
    iedb_iou_threshold: float = DEFAULT_IEDB_IOU_THRESHOLD,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    progress_events: list[dict[str, Any]] = []

    def _emit_progress(
        event: dict[str, Any] | None = None,
        *,
        event_type: str | None = None,
        step_key: str | None = None,
        label: str | None = None,
        status: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any]
        if isinstance(event, dict):
            payload = dict(event)
            payload.setdefault("event_type", "phase_update")
            payload.setdefault("step_key", "unknown")
            payload.setdefault("label", str(payload.get("step_key", "unknown")))
            payload.setdefault("status", "running")
            payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        else:
            payload = {
                "event_type": str(event_type or "phase_update"),
                "step_key": str(step_key or "unknown"),
                "label": str(label or step_key or "unknown"),
                "status": str(status or "running"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            if details is not None:
                payload["details"] = details
        progress_events.append(payload)
        if len(progress_events) > 500:
            del progress_events[:-500]
        if progress_callback is not None:
            try:
                progress_callback(payload)
            except Exception:
                pass

    _emit_progress(
        event_type="phase_start",
        step_key="input_normalization",
        label="Normalize input sequence",
        status="running",
    )
    sequence = normalize_sequence(raw_sequence)
    _emit_progress(
        event_type="phase_done",
        step_key="input_normalization",
        label="Normalize input sequence",
        status="ok",
        details={"sequence_length": len(sequence)},
    )
    runtime_deprecation_warnings: list[str] = []
    runtime_option_warnings: list[str] = []
    input_candidate_source = str(candidate_source or "").strip().lower()
    candidate_source = _normalize_candidate_source(candidate_source)
    if input_candidate_source in {"deterministic", "llm_propose_full"}:
        runtime_deprecation_warnings.append(f"candidate_source_deprecated_alias:{input_candidate_source}->llm_propose")
    if str(failure_policy or DEFAULT_FAILURE_POLICY).strip().lower() != "raw_llm_only":
        runtime_deprecation_warnings.append("failure_policy_deprecated_forced_to_raw_llm_only")
    if str(orchestrator_mode or "react").strip().lower() != "react":
        runtime_deprecation_warnings.append("orchestrator_mode_deprecated_forced_to_react")
    if int(_safe_float(react_max_retries, 1.0)) != 1:
        runtime_deprecation_warnings.append("react_max_retries_deprecated_forced_to_1")
    failure_policy = "raw_llm_only"
    report_view = "compact"
    use_multi_agent = True
    orchestrator_mode = "react"
    react_max_retries = 1
    output_path = Path(output_dir)
    output_base = output_path if output_path.is_absolute() else REPO_ROOT / output_path
    iedb_table = Path(iedb_table_path) if iedb_table_path else DEFAULT_IEDB_TABLE_PATH
    if not iedb_table.is_absolute():
        iedb_table = REPO_ROOT / iedb_table
    top_k = max(1, int(top_k))
    self_consistency_k = max(1, int(self_consistency_k))
    candidate_pool_size = max(top_k, int(candidate_pool_size))
    proposal_target_count = min(top_k + 5, 20)
    primary_max_tokens = min(int(max_tokens), COMPACT_PROPOSER_MAX_TOKENS)
    orchestrator = LightweightReActOrchestrator(
        max_steps=react_max_steps,
        max_retries=react_max_retries,
        step_callback=_emit_progress,
    )

    selected_llm_provider = str(llm_provider or DEFAULT_LLM_PROVIDER or "tinker").strip().lower()
    if selected_llm_provider not in {"tinker", "openrouter"}:
        raise ValueError("llm_provider must be one of: tinker, openrouter")

    if require_api_key:
        if selected_llm_provider == "openrouter":
            if not ensure_openrouter_api_key(REPO_ROOT):
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Run ./scripts/setup_openrouter_key.sh "
                    "and source .openrouter.env, or export OPENROUTER_API_KEY."
                )
        elif not ensure_tinker_api_key(REPO_ROOT):
            raise RuntimeError("TINKER_API_KEY is not set. Run ./scripts/setup_tinker_key.sh and source .tinker.env")

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{_sanitize_label(uniprot)}"
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    selected_base_model = str(base_model).strip() or BASE_MODEL
    selected_checkpoint = str(checkpoint).strip() if checkpoint is not None else ""
    selected_openrouter_model = (
        str(openrouter_model or "").strip()
        or os.environ.get("SITE4DRUG_OPENROUTER_MODEL", "").strip()
        or os.environ.get("OPENROUTER_MODEL", "").strip()
        or OPENROUTER_MODEL_FALLBACK
    )
    selected_openrouter_base_url = (
        str(openrouter_base_url or "").strip()
        or os.environ.get("SITE4DRUG_OPENROUTER_BASE_URL", "").strip()
        or os.environ.get("OPENROUTER_BASE_URL", "").strip()
        or DEFAULT_OPENROUTER_BASE_URL
    )
    selected_openrouter_referer = (
        str(openrouter_referer or "").strip()
        or os.environ.get("SITE4DRUG_OPENROUTER_REFERER", "").strip()
        or os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    )
    selected_openrouter_title = (
        str(openrouter_title or "").strip()
        or os.environ.get("SITE4DRUG_OPENROUTER_TITLE", "").strip()
        or os.environ.get("OPENROUTER_TITLE", "").strip()
        or DEFAULT_OPENROUTER_APP_TITLE
    )

    if selected_llm_provider == "tinker":
        import tinker
        from tinker_cookbook import renderers, tokenizer_utils

        tokenizer = tokenizer_utils.get_tokenizer(selected_base_model)
        renderer = renderers.get_renderer("qwen3", tokenizer)
        sampling_seed_is_supported = sampling_seed_supported(tinker.types)
        if sampling_seed is not None and not sampling_seed_is_supported:
            runtime_option_warnings.append("sampling_seed_requested_but_unsupported")
        service_client = tinker.ServiceClient()
        if selected_checkpoint:
            sampling_client = service_client.create_sampling_client(model_path=selected_checkpoint)
            model_source = "checkpoint"
        else:
            sampling_client = service_client.create_sampling_client(base_model=selected_base_model)
            model_source = "base_model"

        repair_sampling_client = sampling_client
        repair_model_source = model_source
        repair_client_error: str | None = None
        if repair_with_base_model and selected_checkpoint:
            try:
                repair_sampling_client = service_client.create_sampling_client(base_model=selected_base_model)
                repair_model_source = "base_model"
                orchestrator.record(
                    plan="Configure repair-model routing.",
                    execution="create_sampling_client(base_model=selected_base_model)",
                    observation=f"repair_model_source={repair_model_source}",
                    status="ok",
                )
            except Exception as exc:
                repair_sampling_client = sampling_client
                repair_model_source = model_source
                repair_client_error = str(exc)
                orchestrator.record(
                    plan="Configure repair-model routing.",
                    execution="create_sampling_client(base_model=selected_base_model)",
                    observation="repair model setup failed; using primary model for repair",
                    status="warn",
                    error_code="repair_model_setup_failed",
                )

        panel_sampling_client = sampling_client
        panel_model_source = model_source
        panel_client_error: str | None = None
        if panel_with_base_model and selected_checkpoint:
            try:
                panel_sampling_client = service_client.create_sampling_client(base_model=selected_base_model)
                panel_model_source = "base_model"
                orchestrator.record(
                    plan="Configure panel-model routing.",
                    execution="create_sampling_client(base_model=selected_base_model)",
                    observation=f"panel_model_source={panel_model_source}",
                    status="ok",
                )
            except Exception as exc:
                panel_sampling_client = sampling_client
                panel_model_source = model_source
                panel_client_error = str(exc)
                orchestrator.record(
                    plan="Configure panel-model routing.",
                    execution="create_sampling_client(base_model=selected_base_model)",
                    observation="panel model setup failed; using primary model for panel",
                    status="warn",
                    error_code="panel_model_setup_failed",
                )
    else:
        tokenizer = None
        renderer = ApproxChatRenderer()
        sampling_seed_is_supported = True
        if selected_checkpoint:
            runtime_option_warnings.append("checkpoint_ignored_for_openrouter_provider")
        if repair_with_base_model:
            runtime_option_warnings.append("repair_with_base_model_uses_openrouter_primary_client")
        if panel_with_base_model:
            runtime_option_warnings.append("panel_with_base_model_uses_openrouter_primary_client")
        sampling_client = OpenRouterChatClient(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            model=selected_openrouter_model,
            base_url=selected_openrouter_base_url,
            referer=selected_openrouter_referer,
            title=selected_openrouter_title,
            timeout=float(openrouter_timeout),
        )
        model_source = "openrouter"
        selected_checkpoint = ""
        repair_sampling_client = sampling_client
        repair_model_source = "openrouter"
        repair_client_error = None
        panel_sampling_client = sampling_client
        panel_model_source = "openrouter"
        panel_client_error = None
        orchestrator.record(
            plan="Configure OpenRouter inference routing.",
            execution="OpenRouterChatClient(chat/completions)",
            observation=(
                f"openrouter_model={selected_openrouter_model or 'account_default'}, "
                f"base_url={selected_openrouter_base_url}"
            ),
            status="ok",
        )

    # Build candidate pools from shared feature engine.
    feature_backend_error: str | None = None
    seq_summary: dict[str, Any] = {
        "sequence_length": len(sequence),
        "hydropathy_profile": [],
        "tm_regions": [],
        "ptm_sites": [],
        "ptm_summary": {},
        "motif_hits": [],
        "motif_summary": {},
        "cysteine_positions": [],
        "feature_provenance": {},
    }
    candidates: list[dict[str, Any]] = []
    _emit_progress(
        event_type="phase_start",
        step_key="feature_extraction",
        label="Extract constraint features and candidate pool",
        status="running",
    )
    try:
        if mode == "auto":
            epi = build_candidate_features(
                sequence,
                mode="epitope",
                top_n=max(candidate_pool_size // 2, top_k * 3),
                ptm_source=ptm_source,
                ptm_policy=ptm_policy,
                motif_source=motif_source,
                use_motif=use_motif,
                musitedeep_api_base_url=musitedeep_api_base_url,
                musitedeep_model_map=musitedeep_model_map,
                strict_ptm_backend=str(ptm_source).strip().lower() == "musitedeep",
            )
            poc = build_candidate_features(
                sequence,
                mode="pocket",
                top_n=max(candidate_pool_size // 2, top_k * 3),
                ptm_source=ptm_source,
                ptm_policy=ptm_policy,
                motif_source=motif_source,
                use_motif=use_motif,
                musitedeep_api_base_url=musitedeep_api_base_url,
                musitedeep_model_map=musitedeep_model_map,
                strict_ptm_backend=str(ptm_source).strip().lower() == "musitedeep",
            )
            seq_summary = {
                "sequence_length": len(sequence),
                "hydropathy_profile": epi["hydropathy_profile"],
                "tm_regions": epi["tm_regions"],
                "ptm_sites": epi["ptm_sites"],
                "ptm_summary": epi.get("ptm_summary", {}),
                "motif_hits": epi.get("motif_hits", []),
                "motif_summary": epi.get("motif_summary", {}),
                "cysteine_positions": epi["cysteine_positions"],
                "feature_provenance": epi.get("feature_provenance", {}),
                "ptm_backend": epi.get("ptm_backend", {}),
                "api_raw": epi.get("api_raw", {}),
            }
            for c in epi["candidates"]:
                cc = dict(c)
                cc["candidate_id"] = f"E_{c['candidate_id']}"
                cc["mode"] = "epitope"
                candidates.append(cc)
            for c in poc["candidates"]:
                cc = dict(c)
                cc["candidate_id"] = f"P_{c['candidate_id']}"
                cc["mode"] = "pocket"
                candidates.append(cc)
            candidates = candidates[:candidate_pool_size]
        else:
            bundle = build_candidate_features(
                sequence,
                mode=mode,
                top_n=candidate_pool_size,
                ptm_source=ptm_source,
                ptm_policy=ptm_policy,
                motif_source=motif_source,
                use_motif=use_motif,
                musitedeep_api_base_url=musitedeep_api_base_url,
                musitedeep_model_map=musitedeep_model_map,
                strict_ptm_backend=str(ptm_source).strip().lower() == "musitedeep",
            )
            seq_summary = {
                "sequence_length": len(sequence),
                "hydropathy_profile": bundle["hydropathy_profile"],
                "tm_regions": bundle["tm_regions"],
                "ptm_sites": bundle["ptm_sites"],
                "ptm_summary": bundle.get("ptm_summary", {}),
                "motif_hits": bundle.get("motif_hits", []),
                "motif_summary": bundle.get("motif_summary", {}),
                "cysteine_positions": bundle["cysteine_positions"],
                "feature_provenance": bundle.get("feature_provenance", {}),
                "ptm_backend": bundle.get("ptm_backend", {}),
                "api_raw": bundle.get("api_raw", {}),
            }
            candidates = bundle["candidates"]
    except PTMBackendError as exc:
        feature_backend_error = str(exc)
        seq_summary["feature_provenance"] = {
            "ptm_source": ptm_source,
            "motif_source": motif_source,
            "motif_remote_status": "not_requested",
            "ptm_backend_effective": "failed",
            "musitedeep_status": "failed",
            "musitedeep_error_summary": feature_backend_error,
        }
        seq_summary["api_raw"] = {
            "musitedeep": {"status": "failed", "errors": feature_backend_error, "raw_calls": []},
            "scanprosite": {"status": "not_requested", "raw_xml": "", "error": "", "warning": "", "n_hits": 0},
        }

    orchestrator.record(
        plan="Compute sequence constraints and candidate pool.",
        execution=(
            f"build_candidate_features(mode={mode}, ptm_source={ptm_source}, "
            f"ptm_policy={ptm_policy}, motif_source={motif_source}, use_motif={use_motif})"
        ),
        observation=(
            f"candidates={len(candidates)}, tm_regions={len(seq_summary.get('tm_regions', []))}, "
            f"ptm_sites={len(seq_summary.get('ptm_sites', []))}, motifs={len(seq_summary.get('motif_hits', []))}"
        )
        if not feature_backend_error
        else f"feature_backend_failure={feature_backend_error}",
        status="ok" if not feature_backend_error else "warn",
        error_code=None if not feature_backend_error else "feature_backend_failed",
    )
    _emit_progress(
        event_type="phase_done",
        step_key="feature_extraction",
        label="Extract constraint features and candidate pool",
        status="ok" if not feature_backend_error else "warn",
        details={
            "candidates": len(candidates),
            "tm_regions": len(seq_summary.get("tm_regions", [])),
            "ptm_sites": len(seq_summary.get("ptm_sites", [])),
            "motif_hits": len(seq_summary.get("motif_hits", [])),
            "feature_backend_error": feature_backend_error or "",
        },
    )
    auto_mode_policy = _auto_mode_policy(seq_summary) if mode == "auto" else None
    feature_warnings: list[str] = []
    ptm_backend_meta = seq_summary.get("ptm_backend", {}) if isinstance(seq_summary, dict) else {}
    if isinstance(ptm_backend_meta, dict):
        for w in ptm_backend_meta.get("warnings", []) or []:
            feature_warnings.append(f"ptm_backend:{w}")
    motif_remote_status = str((seq_summary.get("feature_provenance", {}) or {}).get("motif_remote_status", ""))
    if str(motif_source).strip().lower() in {"remote", "auto"} and "no_hits" in motif_remote_status:
        feature_warnings.append(f"motif_remote_status:{motif_remote_status}")
    if auto_mode_policy:
        orchestrator.record(
            plan="Apply deterministic auto-modality policy.",
            execution="classify non-membrane/channel-like/general membrane from TM+motif evidence",
            observation=(
                f"auto_mode={auto_mode_policy.get('mode')}, "
                f"n_tm_regions={auto_mode_policy.get('n_tm_regions')}, "
                f"channel_like={auto_mode_policy.get('channel_like')}"
            ),
            status="ok",
        )

    token_strategy_used = "full_sequence"
    token_events: list[dict[str, Any]] = []
    active_messages: list[dict] = []
    active_raw_text = ""
    active_candidates = candidates

    if feature_backend_error:
        token_strategy_used = "feature_backend_failure"
        active_candidates = []
        orchestrator.record(
            plan="Skip LLM generation due feature backend failure.",
            execution="Bypass prompt build/sample path.",
            observation=f"feature_backend_failure={feature_backend_error}",
            status="warn",
            error_code="feature_backend_failed",
        )
    else:
        messages_full = _build_full_sequence_messages(
            uniprot=uniprot,
            sequence=sequence,
            mode=mode,
            top_k=proposal_target_count,
            seq_summary=seq_summary,
            candidate_source=candidate_source,
            auto_mode_policy=auto_mode_policy,
        )
        full_budget = evaluate_budget(
            strategy="full_sequence",
            renderer=renderer,
            messages=messages_full,
            max_input_tokens=max_input_tokens,
        )
        token_events = [full_budget.__dict__]
        active_messages = messages_full
        orchestrator.record(
            plan="Select initial prompt strategy.",
            execution="Evaluate token budget for compact full-sequence proposal prompt.",
            observation=f"strategy=full_sequence, within_budget={full_budget.is_within_budget}, tokens={full_budget.input_tokens}",
            status="ok",
        )
        if not full_budget.is_within_budget:
            token_strategy_used = "input_overflow"
            orchestrator.record(
                plan="Overflow handling.",
                execution="Skip LLM sample when prompt exceeds configured input budget.",
                observation="run_status will preserve raw output diagnostics without deterministic fallback",
                status="warn",
                error_code="token_budget_exceeded",
            )

    started = time.time()
    parser_meta: dict[str, Any] = {}
    proposal_stats: dict[str, Any] = {
        "llm_proposal_total": 0,
        "llm_proposal_valid": 0,
        "llm_proposal_dropped": 0,
        "llm_proposal_fill_count": 0,
        "proposal_validation_errors": [],
        "llm_valid_rows": [],
    }
    self_consistency_enabled = bool(_is_llm_candidate_source(candidate_source) and self_consistency_k > 1)
    self_consistency_meta: dict[str, Any] = {
        "enabled": bool(self_consistency_enabled),
        "requested_k": int(self_consistency_k),
        "effective_k": 1,
        "successful_attempts": 0,
        "consensus_status": "disabled",
        "selected_attempt_index": None,
        "modality_votes": {"epitope": 0, "pocket": 0, "other": 0},
        "attempts": [],
        "final_candidate_votes": [],
    }
    validation_context: dict[str, Any] = {
        "allow_missing_candidate_id": _is_llm_candidate_source(candidate_source),
        "proposal_target_count": proposal_target_count,
        "ptm_sites": seq_summary.get("ptm_sites", []),
        "motif_hits": seq_summary.get("motif_hits", []),
    }
    parse_mode = "proposal"

    def _empty_proposal_stats() -> dict[str, Any]:
        return {
            "llm_proposal_total": 0,
            "llm_proposal_valid": 0,
            "llm_proposal_dropped": 0,
            "llm_proposal_fill_count": 0,
            "proposal_validation_errors": [],
            "llm_valid_rows": [],
        }

    proposal_attempts: list[dict[str, Any]] = []
    attempts_by_index: dict[int, dict[str, Any]] = {}

    def _run_proposal_attempt(attempt_index: int) -> dict[str, Any]:
        attempt_started = time.time()
        attempt_raw_text = _sample_text(
            sampling_client=sampling_client,
            renderer=renderer,
            tokenizer=tokenizer,
            messages=active_messages,
            max_tokens=primary_max_tokens,
            temperature=temperature,
            sampling_seed=_derived_sampling_seed(sampling_seed, stage_offset=0, attempt_index=attempt_index - 1),
        )

        def _repair_fn(repair_prompt: str) -> str:
            repair_messages = list(active_messages) + [
                {"role": "assistant", "content": attempt_raw_text},
                {"role": "user", "content": repair_prompt},
            ]
            return _sample_text(
                sampling_client=repair_sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                messages=repair_messages,
                max_tokens=primary_max_tokens,
                temperature=0.0,
                sampling_seed=_derived_sampling_seed(
                    sampling_seed,
                    stage_offset=100000,
                    attempt_index=attempt_index - 1,
                ),
            )

        attempt_parsed_obj, attempt_parser_meta = parse_with_single_repair(
            attempt_raw_text,
            repair_fn=_repair_fn,
            validation_context=validation_context,
            max_repairs=1,
            parse_mode=parse_mode,
        )
        attempt_parser_meta["repair_model_source"] = repair_model_source
        attempt_parser_meta["repair_with_base_model"] = bool(repair_with_base_model)
        if repair_client_error:
            attempt_parser_meta["repair_model_setup_error"] = repair_client_error

        attempt_site_obj: dict[str, Any] | None = None
        attempt_proposal_stats = _empty_proposal_stats()
        parsed_for_ranking = deepcopy(attempt_parsed_obj) if isinstance(attempt_parsed_obj, dict) else None
        if parsed_for_ranking is not None:
            if mode == "auto" and auto_mode_policy:
                parsed_for_ranking["recommended_modality"] = str(auto_mode_policy.get("mode", "epitope"))
                parsed_for_ranking["modality_confidence"] = float(_safe_float(auto_mode_policy.get("confidence"), 0.85))
                attempt_parser_meta["auto_mode_policy_applied"] = True
                attempt_parser_meta["auto_mode_policy_mode"] = parsed_for_ranking["recommended_modality"]
                attempt_parser_meta["auto_mode_policy_reason"] = str(auto_mode_policy.get("reason", ""))
            if _is_llm_candidate_source(candidate_source):
                attempt_site_obj, attempt_proposal_stats = _validate_and_enrich_llm_proposals(
                    parsed_obj=parsed_for_ranking,
                    sequence=sequence,
                    requested_mode=mode,
                    top_k=top_k,
                    seq_summary=seq_summary,
                    deterministic_candidates=active_candidates,
                    ptm_policy=ptm_policy,
                    allow_fill=False,
                )
                attempt_parser_meta["proposal_validation_errors"] = list(
                    attempt_proposal_stats.get("proposal_validation_errors", [])
                )
                attempt_parser_meta["llm_proposal_total"] = int(attempt_proposal_stats.get("llm_proposal_total", 0))
                attempt_parser_meta["llm_proposal_valid"] = int(attempt_proposal_stats.get("llm_proposal_valid", 0))
                attempt_parser_meta["llm_proposal_dropped"] = int(
                    attempt_proposal_stats.get("llm_proposal_dropped", 0)
                )
                attempt_parser_meta["llm_proposal_fill_count"] = int(
                    attempt_proposal_stats.get("llm_proposal_fill_count", 0)
                )
                attempt_site_obj["llm_generated_candidates"] = deepcopy(
                    attempt_proposal_stats.get("llm_valid_rows", [])
                )
            else:
                candidate_lookup = {c["candidate_id"]: c for c in active_candidates}
                attempt_site_obj = _inject_ranked_candidate_details(parsed_for_ranking, candidate_lookup, top_k=top_k)

        attempt_ranked = list((attempt_site_obj or {}).get("ranked_candidates", []) or [])
        attempt_preview: list[dict[str, Any]] = []
        for candidate in attempt_ranked[:5]:
            reason = str(candidate.get("reason", "") or "").strip()
            attempt_preview.append(
                {
                    "rank": int(_safe_float(candidate.get("rank"), 0)),
                    "candidate_id": str(candidate.get("candidate_id", "") or ""),
                    "mode": str(candidate.get("mode", "") or ""),
                    "start": int(_safe_float(candidate.get("start"), 0)),
                    "end": int(_safe_float(candidate.get("end"), 0)),
                    "peptide": str(candidate.get("peptide", "") or ""),
                    "confidence_score": round(_safe_float(candidate.get("confidence_score"), 0.0), 3),
                    "flags": [str(flag) for flag in list(candidate.get("flags", []) or [])[:8]],
                    "reason": reason[:220],
                }
            )
        attempt_elapsed_seconds = max(time.time() - attempt_started, 0.0)
        return {
            "attempt_index": int(attempt_index),
            "raw_text": attempt_raw_text,
            "parsed_obj": deepcopy(attempt_parsed_obj) if isinstance(attempt_parsed_obj, dict) else None,
            "parser_meta": dict(attempt_parser_meta),
            "proposal_stats": deepcopy(attempt_proposal_stats),
            "site_obj": deepcopy(attempt_site_obj) if isinstance(attempt_site_obj, dict) else None,
            "ranked_candidates": deepcopy(attempt_ranked),
            "summary": {
                "attempt_index": int(attempt_index),
                "parser_status": str(attempt_parser_meta.get("parser_status", "unknown")),
                "parser_errors": list(attempt_parser_meta.get("parser_errors", [])),
                "parsed_candidate_count": len(attempt_ranked),
                "recommended_modality": (
                    str(attempt_parsed_obj.get("recommended_modality"))
                    if isinstance(attempt_parsed_obj, dict)
                    else None
                ),
                "llm_proposal_total": int(attempt_proposal_stats.get("llm_proposal_total", 0)),
                "llm_proposal_valid": int(attempt_proposal_stats.get("llm_proposal_valid", 0)),
                "llm_proposal_dropped": int(attempt_proposal_stats.get("llm_proposal_dropped", 0)),
                "llm_proposal_fill_count": int(attempt_proposal_stats.get("llm_proposal_fill_count", 0)),
                "raw_output_chars": len(attempt_raw_text),
                "elapsed_seconds": round(float(attempt_elapsed_seconds), 3),
                "ranked_candidates_preview": attempt_preview,
            },
        }

    if token_strategy_used not in {"input_overflow", "feature_backend_failure"}:
        effective_self_consistency_k = int(self_consistency_k) if self_consistency_enabled else 1
        self_consistency_meta["effective_k"] = effective_self_consistency_k
        _emit_progress(
            event_type="phase_start",
            step_key="model_sampling",
            label="Generate model output",
            status="running",
            details={
                "token_strategy": token_strategy_used,
                "self_consistency_enabled": bool(self_consistency_enabled),
                "attempts": effective_self_consistency_k,
            },
        )
        orchestrator.record(
            plan="Generate model output from selected prompt strategy.",
            execution=(
                f"sample(prompt_strategy={token_strategy_used}, max_tokens={max_tokens}, "
                f"temperature={temperature}, attempts={effective_self_consistency_k})"
            ),
            observation="Sampling started.",
            status="ok",
        )
        _emit_progress(
            event_type="phase_start",
            step_key="parse_and_validate",
            label="Parse and validate model output",
            status="running",
        )
        proposal_attempts = [_run_proposal_attempt(attempt_index) for attempt_index in range(1, effective_self_consistency_k + 1)]
        attempts_by_index = {
            int(attempt.get("attempt_index", 0)): attempt
            for attempt in proposal_attempts
        }
        successful_attempts = [attempt for attempt in proposal_attempts if attempt.get("parsed_obj") is not None]
        if self_consistency_enabled:
            self_consistency_meta["attempts"] = [dict(attempt.get("summary", {})) for attempt in proposal_attempts]
        else:
            self_consistency_meta["successful_attempts"] = len(successful_attempts)
        selected_attempt: dict[str, Any] | None = None
        if self_consistency_enabled:
            consensus = _build_self_consistency_consensus(
                proposal_attempts,
                requested_k=effective_self_consistency_k,
                top_k=top_k,
            )
            self_consistency_meta.update(
                {
                    "successful_attempts": int(consensus.get("successful_attempts", 0)),
                    "consensus_status": str(consensus.get("consensus_status", "no_valid_attempts")),
                    "selected_attempt_index": consensus.get("selected_attempt_index"),
                    "modality_votes": dict(consensus.get("modality_votes", {})),
                    "final_candidate_votes": list(consensus.get("final_candidate_votes", [])),
                }
            )
            selected_idx = consensus.get("selected_attempt_index")
            if selected_idx is not None:
                selected_attempt = attempts_by_index.get(int(selected_idx))
        else:
            consensus = None
            selected_attempt = proposal_attempts[0] if proposal_attempts else None

        if selected_attempt is None:
            ranked_attempts = [attempt for attempt in proposal_attempts if attempt.get("ranked_candidates")]
            if ranked_attempts:
                selected_attempt = min(
                    ranked_attempts,
                    key=lambda attempt: (
                        -len(list(attempt.get("ranked_candidates", []) or [])),
                        _self_consistency_rank_key(
                            {
                                "row": dict((attempt.get("ranked_candidates", []) or [{}])[0]),
                            }
                        ),
                        int(attempt.get("attempt_index", 0)),
                    ),
                )
            elif successful_attempts:
                selected_attempt = successful_attempts[0]
            elif proposal_attempts:
                selected_attempt = proposal_attempts[-1]

        if selected_attempt is not None:
            active_raw_text = str(selected_attempt.get("raw_text", "") or "")
            parsed_obj = deepcopy(selected_attempt.get("parsed_obj"))
            parser_meta = dict(selected_attempt.get("parser_meta", {}))
            proposal_stats = deepcopy(selected_attempt.get("proposal_stats", _empty_proposal_stats()))
            selected_site_obj = deepcopy(selected_attempt.get("site_obj")) if selected_attempt.get("site_obj") else None
        else:
            active_raw_text = ""
            parsed_obj = None
            parser_meta = {"parser_status": "failed"}
            proposal_stats = _empty_proposal_stats()
            selected_site_obj = None

        consensus_ranked = list(consensus.get("ranked_candidates", []) or []) if isinstance(consensus, dict) else []
        if self_consistency_enabled and consensus_ranked:
            base_site_obj = selected_site_obj or {
                "recommended_modality": consensus.get("recommended_modality", "epitope"),
                "modality_confidence": consensus.get("modality_confidence", 0.0),
                "ranked_candidates": [],
                "candidate_evidence": [],
                "risk_flags": [],
                "agent_traces": {},
                "feature_provenance": seq_summary.get("feature_provenance", {}),
                "token_strategy_used": token_strategy_used,
                "audit_log": {"warnings": [], "events": []},
            }
            site_obj = with_schema_defaults(base_site_obj)
            site_obj["recommended_modality"] = str(consensus.get("recommended_modality", "epitope"))
            site_obj["modality_confidence"] = float(_safe_float(consensus.get("modality_confidence"), 0.0))
            site_obj["ranked_candidates"] = deepcopy(consensus_ranked)
            site_obj["llm_generated_candidates"] = deepcopy(consensus_ranked)
            site_obj.setdefault("audit_log", {}).setdefault("warnings", [])
            if str(consensus.get("consensus_status", "disabled")) != "majority":
                site_obj["audit_log"]["warnings"].append(
                    f"self_consistency_status:{consensus.get('consensus_status', 'unknown')}"
                )
        else:
            site_obj = with_schema_defaults(selected_site_obj) if isinstance(selected_site_obj, dict) else None

        parser_meta["repair_model_source"] = repair_model_source
        parser_meta["repair_with_base_model"] = bool(repair_with_base_model)
        parser_meta["self_consistency_enabled"] = bool(self_consistency_enabled)
        parser_meta["self_consistency_k"] = effective_self_consistency_k
        parser_meta["self_consistency_selected_attempt"] = (
            int(selected_attempt.get("attempt_index", 0)) if selected_attempt is not None else None
        )
        if self_consistency_enabled:
            parser_meta["self_consistency_consensus_status"] = str(
                self_consistency_meta.get("consensus_status", "unknown")
            )
            parser_meta["self_consistency_successful_attempts"] = int(
                self_consistency_meta.get("successful_attempts", 0)
            )
        _emit_progress(
            event_type="phase_done",
            step_key="model_sampling",
            label="Generate model output",
            status="ok",
            details={
                "raw_output_chars": len(active_raw_text),
                "attempts": effective_self_consistency_k,
            },
        )
        orchestrator.record(
            plan="Validate LLM output against compact parse rules before enrichment.",
            execution=f"parse_with_single_repair(max_repairs=1, attempts={effective_self_consistency_k})",
            observation=(
                f"parser_status={parser_meta.get('parser_status')}, "
                f"consensus_status={self_consistency_meta.get('consensus_status', 'disabled')}"
            ),
            status="ok" if parsed_obj is not None else "warn",
            error_code=None if parsed_obj is not None else "parse_validation_failed",
        )
        _emit_progress(
            event_type="phase_done",
            step_key="parse_and_validate",
            label="Parse and validate model output",
            status="ok" if parsed_obj is not None else "warn",
            details={
                "parser_status": parser_meta.get("parser_status"),
                "parser_errors": list(parser_meta.get("parser_errors", [])),
                "parse_mode": parser_meta.get("parse_mode", parse_mode),
                "attempts": effective_self_consistency_k,
            },
        )
    else:
        site_obj = None
        parsed_obj = None
        parser_meta = {
            "parser_status": (
                "skipped_due_feature_backend_failure"
                if token_strategy_used == "feature_backend_failure"
                else "skipped_due_token_overflow"
            )
        }
        parser_meta["self_consistency_enabled"] = False
        parser_meta["self_consistency_k"] = 1
        if feature_backend_error:
            parser_meta["parser_errors"] = [feature_backend_error]
        orchestrator.record(
            plan="Bypass LLM generation after pre-checks.",
            execution=(
                "Skip sample+parse due feature backend failure."
                if token_strategy_used == "feature_backend_failure"
                else "Skip sample+parse due configured input-budget overflow."
            ),
            observation=f"parser_status={parser_meta.get('parser_status')}",
            status="warn",
            error_code=("feature_backend_failed" if token_strategy_used == "feature_backend_failure" else "token_budget_exceeded"),
        )
        _emit_progress(
            event_type="phase_done",
            step_key="model_sampling",
            label="Generate model output",
            status="warn",
            details={"skipped": True, "token_strategy": token_strategy_used},
        )
        _emit_progress(
            event_type="phase_done",
            step_key="parse_and_validate",
            label="Parse and validate model output",
            status="warn",
            details={"skipped": True, "parser_status": parser_meta.get("parser_status")},
        )

    run_status = "ok"
    if parsed_obj is None:
        run_status = "failed_feature_backend" if feature_backend_error else "raw_output_only"
        warnings = []
        if feature_backend_error:
            warnings.append(f"feature_backend_failure:{feature_backend_error}")
        parser_errors = list(parser_meta.get("parser_errors", []))
        if not parser_errors and not feature_backend_error:
            parser_errors.append("parse_failed_no_recoverable_candidates")
        if parser_errors:
            warnings.append(f"raw_output_validation_failed:{','.join(parser_errors[:5])}")
        site_obj = with_schema_defaults(
            {
                "recommended_modality": "epitope" if mode == "auto" else mode,
                "modality_confidence": 0.0,
                "ranked_candidates": [],
                "candidate_evidence": [],
                "risk_flags": [],
                "agent_traces": {},
                "feature_provenance": seq_summary.get("feature_provenance", {}),
                "token_strategy_used": token_strategy_used,
                "audit_log": {"warnings": warnings, "events": []},
            }
        )
        orchestrator.record(
            plan="Finalize failure output without deterministic fallback.",
            execution="Build raw-output-only response and preserve parser diagnostics.",
            observation=f"run_status={run_status}",
            status="warn",
            error_code=run_status,
        )
    else:
        if site_obj is None:
            if mode == "auto" and auto_mode_policy:
                parsed_obj["recommended_modality"] = str(auto_mode_policy.get("mode", "epitope"))
                parsed_obj["modality_confidence"] = float(_safe_float(auto_mode_policy.get("confidence"), 0.85))
                parser_meta["auto_mode_policy_applied"] = True
                parser_meta["auto_mode_policy_mode"] = parsed_obj["recommended_modality"]
                parser_meta["auto_mode_policy_reason"] = str(auto_mode_policy.get("reason", ""))
            if _is_llm_candidate_source(candidate_source):
                site_obj, proposal_stats = _validate_and_enrich_llm_proposals(
                    parsed_obj=parsed_obj,
                    sequence=sequence,
                    requested_mode=mode,
                    top_k=top_k,
                    seq_summary=seq_summary,
                    deterministic_candidates=active_candidates,
                    ptm_policy=ptm_policy,
                    allow_fill=False,
                )
                parser_meta["proposal_validation_errors"] = list(proposal_stats.get("proposal_validation_errors", []))
                parser_meta["llm_proposal_total"] = int(proposal_stats.get("llm_proposal_total", 0))
                parser_meta["llm_proposal_valid"] = int(proposal_stats.get("llm_proposal_valid", 0))
                parser_meta["llm_proposal_dropped"] = int(proposal_stats.get("llm_proposal_dropped", 0))
                parser_meta["llm_proposal_fill_count"] = int(proposal_stats.get("llm_proposal_fill_count", 0))
                site_obj["llm_generated_candidates"] = deepcopy(proposal_stats.get("llm_valid_rows", []))
            else:
                candidate_lookup = {c["candidate_id"]: c for c in active_candidates}
                site_obj = _inject_ranked_candidate_details(parsed_obj, candidate_lookup, top_k=top_k)
        if not site_obj.get("ranked_candidates"):
            run_status = "raw_output_only"
            site_obj = with_schema_defaults(
                {
                    "recommended_modality": parsed_obj.get("recommended_modality", "epitope"),
                    "modality_confidence": _safe_float(parsed_obj.get("modality_confidence"), 0.0),
                    "ranked_candidates": [],
                    "candidate_evidence": [],
                    "risk_flags": list(parsed_obj.get("risk_flags", [])),
                    "agent_traces": {},
                    "feature_provenance": seq_summary.get("feature_provenance", {}),
                    "token_strategy_used": token_strategy_used,
                    "audit_log": {
                        "warnings": ["raw_output_only_no_recoverable_candidates"],
                        "events": [],
                    },
                }
            )
    site_obj.setdefault("llm_generated_candidates", [])

    if mode in {"epitope", "pocket"}:
        site_obj["recommended_modality"] = mode

    dropped_count = int(proposal_stats.get("llm_proposal_dropped", 0))
    fill_count = int(proposal_stats.get("llm_proposal_fill_count", 0))
    orchestrator.record(
        plan="Validate and enrich LLM-proposed candidates.",
        execution="bounds/peptide/mode checks + feature enrichment + partial recovery without deterministic fill",
        observation=(
            f"proposed={proposal_stats.get('llm_proposal_total', 0)}, "
            f"valid={proposal_stats.get('llm_proposal_valid', 0)}, "
            f"dropped={dropped_count}, fill={fill_count}"
        ),
        status="ok" if dropped_count == 0 and fill_count == 0 else "warn",
        error_code=None if dropped_count == 0 and fill_count == 0 else "proposal_drop_fill_applied",
    )

    # Multi-agent adjudication
    agent_traces = {}
    agent_warnings: list[str] = []
    panel_comparison: dict[str, Any] = {
        "available": False,
        "used_as_primary": False,
        "panel_status": "not_run",
        "decision_fallback": False,
        "recommended_modality": None,
        "modality_confidence": None,
        "ranked_candidates": [],
    }
    if use_multi_agent and not feature_backend_error:
        base_snapshot = {
            "recommended_modality": site_obj.get("recommended_modality", "epitope"),
            "modality_confidence": float(site_obj.get("modality_confidence", 0.5)),
            "ranked_candidates": deepcopy(site_obj.get("ranked_candidates", [])),
        }
        panel_candidates, panel_candidate_source = _select_panel_candidates(
            candidate_source=candidate_source,
            llm_generated_candidates=site_obj.get("llm_generated_candidates", []),
            deterministic_candidates=active_candidates,
        )
        parser_status = str(parser_meta.get("parser_status", "unknown"))
        llm_json_compatible = parser_status in {"ok", "repaired"}
        if not panel_candidates:
            panel_comparison["panel_status"] = "not_run_empty_candidate_pool"
            agent_warnings.append("multi_agent_skipped_due_empty_candidate_pool")
            orchestrator.record(
                plan="Adjudicate ranking with specialist agent panel.",
                execution="skip multi-agent panel due empty recovered proposal set",
                observation=(
                    f"panel_candidate_source={panel_candidate_source}, panel_candidates={len(panel_candidates)}"
                ),
                status="warn",
                error_code="multi_agent_skipped_empty_candidate_pool",
            )
        else:
            orchestrator.record(
                plan="Adjudicate ranking with specialist agent panel.",
                execution="run_multi_agent_reasoning(BioAgent, ChemAgent, RiskAgent, DecisionAgent)",
                observation=(
                    f"multi-agent panel started (parser_status={parser_status}, "
                    f"panel_candidate_source={panel_candidate_source}, panel_candidates={len(panel_candidates)})"
                ),
                status="ok",
            )
            _emit_progress(
                event_type="phase_start",
                step_key="multi_agent_panel",
                label="Run multi-agent panel",
                status="running",
                details={
                    "candidate_pool_source": panel_candidate_source,
                    "candidate_pool_size": len(panel_candidates),
                },
            )
            decision, traces = run_multi_agent_reasoning(
                sampling_client=panel_sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                sequence_summary=seq_summary,
                candidates=panel_candidates,
                requested_mode=mode,
                top_k=top_k,
                deterministic_only=False,
                progress_callback=_emit_progress,
                sampling_seed=_derived_sampling_seed(sampling_seed, stage_offset=200000),
            )
            if not llm_json_compatible:
                agent_warnings.append("base_output_non_json_multi_agent_attempted")
            if panel_client_error:
                agent_warnings.append(f"panel_model_setup_error:{panel_client_error}")
            ranking = decision.get("ranking", [])
            decision_fallback = bool(traces.get("decision_agent", {}).get("fallback"))
            panel_status = traces.get("panel_status", "unknown")
            panel_modality = decision.get("recommended_modality", base_snapshot["recommended_modality"])
            panel_confidence = float(decision.get("modality_confidence", base_snapshot["modality_confidence"]))
            candidate_lookup = {
                str(c.get("candidate_id")): c
                for c in panel_candidates
                if str(c.get("candidate_id", "")).strip()
            }
            panel_ranked = _materialize_panel_ranking(
                ranking,
                candidate_lookup,
                top_k=top_k,
                default_modality_confidence=panel_confidence,
                decision_fallback=decision_fallback,
                fallback_mode=str(panel_modality or base_snapshot["recommended_modality"]),
            )
            authoritative_mode = str(base_snapshot["recommended_modality"] or "other").strip().lower()
            if authoritative_mode in {"epitope", "pocket"}:
                filtered_panel_ranked = [
                    row for row in panel_ranked if str(row.get("mode", "")).strip().lower() == authoritative_mode
                ]
                if filtered_panel_ranked != panel_ranked:
                    agent_warnings.append("decision_panel_filtered_to_authoritative_modality")
                panel_ranked = filtered_panel_ranked
            apply_panel_override = panel_status == "ok" and bool(panel_ranked)
            panel_comparison = {
                "available": bool(panel_ranked),
                "used_as_primary": bool(apply_panel_override),
                "panel_status": panel_status,
                "decision_fallback": bool(decision_fallback),
                "recommended_modality": panel_modality,
                "modality_confidence": panel_confidence,
                "candidate_pool_source": panel_candidate_source,
                "candidate_pool_size": len(panel_candidates),
                "ranked_candidates": panel_ranked,
            }
            if apply_panel_override:
                site_obj["recommended_modality"] = base_snapshot["recommended_modality"]
                if str(panel_modality).strip().lower() == authoritative_mode:
                    site_obj["modality_confidence"] = panel_confidence
                else:
                    site_obj["modality_confidence"] = base_snapshot["modality_confidence"]
                    agent_warnings.append(
                        f"decision_panel_modality_disagreement:{panel_modality}->{base_snapshot['recommended_modality']}"
                    )
                site_obj["ranked_candidates"] = panel_ranked
            else:
                site_obj["recommended_modality"] = base_snapshot["recommended_modality"]
                site_obj["modality_confidence"] = base_snapshot["modality_confidence"]
                site_obj["ranked_candidates"] = deepcopy(base_snapshot["ranked_candidates"])
                if panel_status != "ok":
                    agent_warnings.append("multi_agent_primary_preserved_due_panel_status")
                elif not panel_ranked:
                    agent_warnings.append("multi_agent_primary_preserved_due_empty_decision_ranking")
            agent_traces = traces
            if panel_status != "ok":
                agent_warnings.append(f"multi_agent_panel_status:{panel_status}")
            orchestrator.record(
                plan="Observe panel outcome and apply ranking override if valid.",
                execution="merge decision payload into site output",
                observation=(
                    f"panel_status={panel_status}, decision_fallback={bool(traces.get('decision_agent', {}).get('fallback'))}, "
                    f"override_applied={apply_panel_override}, comparison_rows={len(panel_ranked)}"
                ),
                status="ok" if apply_panel_override else "warn",
                error_code=None if apply_panel_override else "multi_agent_primary_preserved",
            )
            for agent_name in ("bio_agent", "chem_agent", "risk_agent", "decision_agent"):
                trace = traces.get(agent_name, {})
                parse_error = trace.get("parse_error")
                if parse_error:
                    agent_warnings.append(f"{agent_name}_parse_error:{parse_error}")
                validation_errors = trace.get("validation_errors", [])
                if validation_errors:
                    joined = ",".join(str(v) for v in validation_errors)
                    agent_warnings.append(f"{agent_name}_validation_errors:{joined}")
            if traces.get("decision_agent", {}).get("fallback"):
                agent_warnings.append("decision_agent_fallback_used")
            _emit_progress(
                event_type="phase_done",
                step_key="multi_agent_panel",
                label="Run multi-agent panel",
                status="ok" if panel_status == "ok" else "warn",
                details={
                    "panel_status": panel_status,
                    "decision_fallback": bool(decision_fallback),
                    "override_applied": bool(apply_panel_override),
                },
            )
    if mode == "auto" and auto_mode_policy:
        policy_mode = str(auto_mode_policy.get("mode", "epitope"))
        if site_obj.get("recommended_modality") != policy_mode:
            agent_warnings.append(
                f"auto_mode_policy_override:{site_obj.get('recommended_modality')}->{policy_mode}"
            )
        site_obj["recommended_modality"] = policy_mode
        site_obj["modality_confidence"] = float(_safe_float(auto_mode_policy.get("confidence"), 0.85))
    site_obj["agent_traces"] = agent_traces
    site_obj["panel_comparison"] = panel_comparison
    candidate_reason_lookup = {
        str(c.get("candidate_id")): dict(c)
        for c in active_candidates
        if str(c.get("candidate_id", "")).strip()
    }
    site_obj["ranked_candidates"] = _ensure_candidate_reasons(
        list(site_obj.get("ranked_candidates", []) or []),
        candidate_lookup=candidate_reason_lookup,
    )
    if isinstance(site_obj.get("panel_comparison"), dict):
        panel_rows = list((site_obj.get("panel_comparison", {}) or {}).get("ranked_candidates", []) or [])
        site_obj["panel_comparison"]["ranked_candidates"] = _ensure_candidate_reasons(
            panel_rows,
            candidate_lookup=candidate_reason_lookup,
        )
    site_obj["llm_generated_candidates"] = _ensure_candidate_reasons(
        list(site_obj.get("llm_generated_candidates", []) or []),
        candidate_lookup=candidate_reason_lookup,
    )
    if feature_warnings:
        site_obj.setdefault("audit_log", {}).setdefault("warnings", []).extend(feature_warnings)
    if runtime_deprecation_warnings:
        site_obj.setdefault("audit_log", {}).setdefault("warnings", []).extend(runtime_deprecation_warnings)
    if runtime_option_warnings:
        site_obj.setdefault("audit_log", {}).setdefault("warnings", []).extend(runtime_option_warnings)
    if agent_warnings:
        site_obj.setdefault("audit_log", {}).setdefault("warnings", []).extend(agent_warnings)

    raw_output_validation = _validate_raw_output_consistency(
        active_raw_text,
        sequence_length=len(sequence),
    )
    if raw_output_validation.get("issues"):
        issue_preview = ",".join(str(x) for x in raw_output_validation["issues"][:8])
        site_obj.setdefault("audit_log", {}).setdefault("warnings", []).append(
            f"raw_output_inconsistency_detected:{issue_preview}"
        )
    orchestrator.record(
        plan="Validate raw text consistency with sequence constraints.",
        execution="_validate_raw_output_consistency(raw_text, sequence_length)",
        observation=(
            f"parsed_rows={raw_output_validation.get('parsed_candidate_count', 0)}, "
            f"issues={len(raw_output_validation.get('issues', []))}"
        ),
        status="ok" if not raw_output_validation.get("issues") else "warn",
        error_code=None if not raw_output_validation.get("issues") else "raw_output_inconsistency",
    )

    elapsed = time.time() - started

    # Fill evidence/risk fields from selected candidates.
    candidate_lookup = {str(c["candidate_id"]): c for c in active_candidates}
    for row in site_obj.get("ranked_candidates", []):
        cid = str(row.get("candidate_id", ""))
        if cid:
            candidate_lookup.setdefault(cid, dict(row))
    evidence_rows = []
    all_flags = set(site_obj.get("risk_flags", []))
    parsed_predictions = []
    for c in site_obj.get("ranked_candidates", []):
        cid = c["candidate_id"]
        base = candidate_lookup.get(cid, {})
        flags = c.get("flags", base.get("risk_flags", []))
        all_flags.update(flags)
        evidence_rows.append(
            {
                "candidate_id": cid,
                "mode": base.get("mode", c.get("mode", "epitope")),
                "evidence": [
                    f"mean_hydropathy={base.get('mean_hydropathy', 0.0):.3f}",
                    f"overlaps_tm={bool(base.get('overlaps_tm', False))}",
                    f"overlaps_ptm_mask={bool(base.get('overlaps_ptm_mask', False))}",
                    f"ptm_overlap_by_type={base.get('ptm_overlap_by_type', {})}",
                    f"ptm_density={_safe_float(base.get('ptm_density', 0.0), 0.0):.3f}",
                    f"motif_hit_count={int(_safe_float(base.get('motif_hit_count', 0.0), 0.0))}",
                    f"cysteine_count={int(base.get('cysteine_count', 0))}",
                    f"heuristic_score={base.get('heuristic_score', 0.0):.4f}",
                ],
                "evidence_present": True,
                "external_validation_evidence_present": False,
            }
        )
        parsed_predictions.append(
            {
                "rank": c.get("rank"),
                "epitope": c.get("peptide"),
                "start": c.get("start"),
                "end": c.get("end"),
                "confidence": c.get("confidence"),
                "confidence_score": c.get("confidence_score"),
                "confidence_source": c.get("confidence_source"),
                "flags": flags,
            }
        )

    iedb_validation = {
        "enabled": bool(use_iedb_validation),
        "status": "not_requested",
        "source": "none",
        "n_reference_spans": 0,
        "iou_threshold": float(max(min(iedb_iou_threshold, 1.0), 0.0)),
        "top_k_hit": False,
        "candidate_annotations": [],
    }
    if use_iedb_validation:
        refs, iedb_meta = _load_iedb_reference_spans(uniprot=uniprot, iedb_table_path=iedb_table)
        iedb_validation["status"] = str(iedb_meta.get("status", "unknown"))
        iedb_validation["source"] = str(iedb_meta.get("source", str(iedb_table)))
        iedb_validation["n_reference_spans"] = int(_safe_float(iedb_meta.get("n_reference_spans", len(refs)), 0.0))
        if refs and site_obj.get("ranked_candidates"):
            ann_by_cid = _annotate_iedb_support(
                ranked_candidates=site_obj.get("ranked_candidates", []),
                references=refs,
                iou_threshold=iedb_validation["iou_threshold"],
            )
            top_k_hit = False
            for c in site_obj.get("ranked_candidates", []):
                cid = str(c.get("candidate_id"))
                ann = ann_by_cid.get(cid, {})
                c["iedb_validation"] = ann
                supported = bool(ann.get("supported"))
                top_k_hit = top_k_hit or supported
                iedb_validation["candidate_annotations"].append(
                    {
                        "candidate_id": cid,
                        "supported": supported,
                        "best_iou": float(_safe_float(ann.get("best_iou"), 0.0)),
                        "substring_match": bool(ann.get("substring_match")),
                    }
                )
            iedb_validation["top_k_hit"] = top_k_hit

            for row in evidence_rows:
                cid = str(row.get("candidate_id"))
                ann = ann_by_cid.get(cid, {})
                supported = bool(ann.get("supported"))
                row["external_validation_evidence_present"] = supported
                row["evidence"].append(f"iedb_supported={supported}")
                row["evidence"].append(f"iedb_best_iou={_safe_float(ann.get('best_iou'), 0.0):.3f}")
                row["evidence"].append(f"iedb_substring_match={bool(ann.get('substring_match'))}")
        elif iedb_validation["status"] == "ok":
            iedb_validation["status"] = "ok_no_match_for_uniprot"

    site_obj["candidate_evidence"] = evidence_rows
    site_obj["risk_flags"] = sorted(all_flags)
    site_obj["ptm_summary"] = seq_summary.get("ptm_summary", {})
    site_obj["motif_summary"] = seq_summary.get("motif_summary", {})
    site_obj["iedb_validation"] = iedb_validation
    merged_provenance = dict(seq_summary.get("feature_provenance", {}) or {})
    merged_provenance.update(
        {
            "module": "site4drug_inference.common.constraint_features",
            "tm_threshold": 1.6,
            "ptm_mask_pad": 5,
            "hydropathy_window": 19,
            "candidate_pool_size": candidate_pool_size,
            "ptm_source": ptm_source,
            "ptm_rule_version": merged_provenance.get("ptm_rule_version", "rulepack_v1"),
            "motif_source": merged_provenance.get("motif_source", motif_source),
            "motif_library_version": merged_provenance.get("motif_library_version", "scanprosite_biopython_v1"),
            "motif_remote_status": merged_provenance.get("motif_remote_status", "not_requested"),
            "orchestrator_mode": orchestrator_mode,
        }
    )
    site_obj["feature_provenance"] = merged_provenance
    site_obj["token_strategy_used"] = token_strategy_used
    site_obj["orchestrator_trace"] = orchestrator.to_list()
    site_obj = with_schema_defaults(site_obj)
    final_validation_errors: list[str] = []
    if run_status in {"ok", "ok_partial"}:
        final_validation_errors = validate_site_output(
            site_obj,
            validation_context={
                "ptm_sites": seq_summary.get("ptm_sites", []),
                "motif_hits": seq_summary.get("motif_hits", []),
            },
        )
        if final_validation_errors:
            run_status = "failed_validation"
            site_obj.setdefault("audit_log", {}).setdefault("warnings", []).append(
                "final_schema_validation_failed:" + ",".join(str(err) for err in final_validation_errors[:8])
            )
    if run_status == "ok" and 0 < len(site_obj.get("ranked_candidates", [])) < top_k:
        run_status = "ok_partial"
    if run_status == "ok" and feature_backend_error:
        run_status = "failed_feature_backend"
    if run_status == "ok" and not site_obj.get("ranked_candidates"):
        run_status = "failed_validation"
    if run_status != "ok":
        _emit_progress(
            event_type="run_error",
            step_key="run_status",
            label="Run status indicates failure/warning",
            status="error",
            details={"run_status": run_status},
        )

    # Persist per-agent traces for auditability.
    agent_artifacts: dict[str, str] = {}
    if agent_traces:
        agent_dir = run_dir / "agents"
        agent_dir.mkdir(parents=True, exist_ok=True)
        for agent_name, trace_payload in agent_traces.items():
            if not str(agent_name).endswith("_agent"):
                continue
            agent_file = agent_dir / f"{agent_name}.json"
            agent_file.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
            agent_artifacts[agent_name] = str(Path("agents") / agent_file.name)
        combined_file = run_dir / "agent_traces.json"
        combined_file.write_text(json.dumps(agent_traces, indent=2), encoding="utf-8")
        agent_artifacts["combined"] = combined_file.name
    orchestrator_file = run_dir / "orchestrator_trace.json"
    orchestrator_file.write_text(json.dumps(orchestrator.to_list(), indent=2), encoding="utf-8")

    # Plot artifacts
    plot_artifacts = {}
    if enable_plot:
        plot_png = run_dir / "hydropathy_ptm_plot.png"
        plot_json = run_dir / "hydropathy_ptm_plot.json"
        try:
            from site4drug_inference.demo.plotting import render_hydropathy_ptm_candidate_plot

            plot_artifacts = render_hydropathy_ptm_candidate_plot(
                sequence=sequence,
                hydropathy_profile=seq_summary.get("hydropathy_profile", []),
                ptm_sites=seq_summary.get("ptm_sites", []),
                ranked_candidates=site_obj.get("ranked_candidates", []),
                output_png=plot_png,
                output_json=plot_json,
                tm_threshold=1.6,
            )
            plot_artifacts["plot_png_name"] = plot_png.name
            plot_artifacts["plot_json_name"] = plot_json.name
            orchestrator.record(
                plan="Render diagnostic visualization artifact.",
                execution="render_hydropathy_ptm_candidate_plot(...)",
                observation="plot_generated=true",
                status="ok",
            )
        except ModuleNotFoundError as exc:
            site_obj.setdefault("audit_log", {}).setdefault("warnings", []).append(
                f"plot_dependency_missing:{exc}"
            )
            orchestrator.record(
                plan="Render diagnostic visualization artifact.",
                execution="render_hydropathy_ptm_candidate_plot(...)",
                observation=f"plot_dependency_missing={exc}",
                status="warn",
                error_code="plot_dependency_missing",
            )
        except Exception as exc:
            site_obj.setdefault("audit_log", {}).setdefault("warnings", []).append(f"plot_generation_failed: {exc}")
            orchestrator.record(
                plan="Render diagnostic visualization artifact.",
                execution="render_hydropathy_ptm_candidate_plot(...)",
                observation=f"plot_generation_failed={exc}",
                status="warn",
                error_code="plot_generation_failed",
            )

    raw_api_calls = _write_raw_api_artifacts(run_dir=run_dir, seq_summary=seq_summary)
    if list(self_consistency_meta.get("attempts", []) or []):
        sc_dir = run_dir / "self_consistency"
        sc_dir.mkdir(parents=True, exist_ok=True)
        attempts_with_artifacts: list[dict[str, Any]] = []
        for attempt in list(self_consistency_meta.get("attempts", []) or []):
            attempt_index = int(_safe_float(attempt.get("attempt_index"), 0.0))
            attempt_raw_text = ""
            matching_attempt = attempts_by_index.get(attempt_index)
            if matching_attempt is not None:
                attempt_raw_text = str(matching_attempt.get("raw_text", "") or "")
            raw_rel = str(Path("self_consistency") / f"attempt_{attempt_index:02d}_raw.txt")
            summary_rel = str(Path("self_consistency") / f"attempt_{attempt_index:02d}_summary.json")
            (run_dir / raw_rel).write_text(attempt_raw_text, encoding="utf-8")
            attempt_with_artifacts = dict(attempt)
            attempt_with_artifacts["raw_artifact_path"] = raw_rel
            attempt_with_artifacts["summary_artifact_path"] = summary_rel
            (run_dir / summary_rel).write_text(json.dumps(attempt_with_artifacts, indent=2), encoding="utf-8")
            attempts_with_artifacts.append(attempt_with_artifacts)
        self_consistency_meta["attempts"] = attempts_with_artifacts
        consensus_rel = str(Path("self_consistency") / "consensus_summary.json")
        (run_dir / consensus_rel).write_text(json.dumps(self_consistency_meta, indent=2), encoding="utf-8")
        self_consistency_meta["consensus_artifact_path"] = consensus_rel

    site_obj["orchestrator_trace"] = orchestrator.to_list()

    run_payload = {
        "run_id": run_id,
        "run_status": run_status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "uniprot": uniprot,
            "sequence_length": len(sequence),
            "sequence": sequence,
            "source": input_source,
            "mode_request": mode,
        },
        "model": {
            "provider": selected_llm_provider,
            "base_model": selected_base_model,
            "checkpoint": selected_checkpoint if selected_checkpoint else None,
            "openrouter_model": selected_openrouter_model if selected_llm_provider == "openrouter" else None,
            "openrouter_base_url": selected_openrouter_base_url if selected_llm_provider == "openrouter" else None,
            "openrouter_referer": selected_openrouter_referer if selected_llm_provider == "openrouter" else None,
            "openrouter_title": selected_openrouter_title if selected_llm_provider == "openrouter" else None,
            "source": model_source,
        },
        "generation": {
            "llm_provider": selected_llm_provider,
            "requested_top_k": top_k,
            "candidate_source": candidate_source,
            "failure_policy": failure_policy,
            "deprecated_option_warnings": list(runtime_deprecation_warnings),
            "option_warnings": list(runtime_option_warnings),
            "max_tokens": max_tokens,
            "primary_max_tokens_effective": primary_max_tokens,
            "temperature": temperature,
            "self_consistency_k": int(self_consistency_k),
            "self_consistency_enabled": bool(self_consistency_enabled),
            "sampling_seed": int(sampling_seed) if sampling_seed is not None else None,
            "sampling_seed_supported": bool(sampling_seed_is_supported),
            "max_input_tokens": max_input_tokens,
            "candidate_pool_size": candidate_pool_size,
            "chunk_size_aa": chunk_size_aa,
            "chunk_overlap_aa": chunk_overlap_aa,
            "orchestrator_mode": orchestrator_mode,
            "react_max_steps": react_max_steps,
            "react_max_retries": react_max_retries,
            "repair_with_base_model": bool(repair_with_base_model),
            "panel_with_base_model": bool(panel_with_base_model),
            "panel_model_source": panel_model_source,
            "ptm_source": ptm_source,
            "ptm_policy": ptm_policy,
            "motif_source": motif_source,
            "musitedeep_api_base_url": musitedeep_api_base_url,
            "musitedeep_model_map": musitedeep_model_map,
            "use_motif": use_motif,
            "use_iedb_validation": use_iedb_validation,
            "iedb_table_path": str(iedb_table),
            "iedb_iou_threshold": iedb_iou_threshold,
            "llm_proposal_total": int(proposal_stats.get("llm_proposal_total", 0)),
            "llm_proposal_valid": int(proposal_stats.get("llm_proposal_valid", 0)),
            "llm_proposal_dropped": int(proposal_stats.get("llm_proposal_dropped", 0)),
            "llm_proposal_fill_count": int(proposal_stats.get("llm_proposal_fill_count", 0)),
            "auto_mode_policy": auto_mode_policy if auto_mode_policy is not None else None,
            "ptm_backend_selected": merged_provenance.get("ptm_backend_selected", ptm_source),
            "ptm_backend_effective": merged_provenance.get("ptm_backend_effective", ptm_source),
            "musitedeep_available": bool(merged_provenance.get("musitedeep_available", False)),
            "musitedeep_status": merged_provenance.get("musitedeep_status", "not_requested"),
            "musitedeep_models_attempted": merged_provenance.get("musitedeep_models_attempted", []),
            "musitedeep_models_succeeded": merged_provenance.get("musitedeep_models_succeeded", []),
            "musitedeep_api_base_url_effective": merged_provenance.get("musitedeep_api_base_url", ""),
            "musitedeep_endpoint_urls": merged_provenance.get("musitedeep_endpoint_urls", []),
            "musitedeep_error_summary": merged_provenance.get("musitedeep_error_summary", ""),
            "motif_source_requested": motif_source,
            "motif_source_effective": merged_provenance.get("motif_source", motif_source),
            "motif_remote_status": merged_provenance.get("motif_remote_status", "not_requested"),
        },
        "timing": {"elapsed_seconds": elapsed},
        "token_strategy_used": token_strategy_used,
        "token_budget_events": token_events,
        "raw_model_output": active_raw_text,
        "raw_output_validation": raw_output_validation,
        "parser_meta": parser_meta,
        "final_validation_errors": final_validation_errors,
        "parsed_predictions": parsed_predictions,  # legacy compatibility
        "parsed_candidate_count": len(parsed_predictions),
        "recommended_modality": site_obj["recommended_modality"],
        "modality_confidence": float(site_obj.get("modality_confidence", 0.0)),
        "ranked_candidates": site_obj.get("ranked_candidates", []),
        "llm_generated_candidates": site_obj.get("llm_generated_candidates", []),
        "candidate_evidence": site_obj.get("candidate_evidence", []),
        "risk_flags": site_obj.get("risk_flags", []),
        "self_consistency": self_consistency_meta,
        "agent_traces": site_obj.get("agent_traces", {}),
        "panel_comparison": site_obj.get("panel_comparison", {}),
        "agent_artifacts": agent_artifacts,
        "orchestrator_artifact": "orchestrator_trace.json",
        "feature_provenance": site_obj.get("feature_provenance", {}),
        "raw_api_calls": raw_api_calls,
        "ptm_summary": site_obj.get("ptm_summary", {}),
        "motif_summary": site_obj.get("motif_summary", {}),
        "iedb_validation": site_obj.get("iedb_validation", {}),
        "orchestrator_trace": orchestrator.to_list(),
        "audit_log": {
            "warnings": list(site_obj.get("audit_log", {}).get("warnings", [])),
            "events": list(site_obj.get("audit_log", {}).get("events", [])),
            "parser_status": parser_meta.get("parser_status", "unknown"),
            "parser_errors": list(parser_meta.get("parser_errors", [])),
            "final_validation_errors": list(final_validation_errors),
            "feature_module": "site4drug_inference.common.constraint_features",
        },
        "proposal_validation": {
            "proposed": int(proposal_stats.get("llm_proposal_total", 0)),
            "valid": int(proposal_stats.get("llm_proposal_valid", 0)),
            "dropped": int(proposal_stats.get("llm_proposal_dropped", 0)),
            "fill_count": int(proposal_stats.get("llm_proposal_fill_count", 0)),
            "errors": list(proposal_stats.get("proposal_validation_errors", [])),
        },
        "plot_artifacts": plot_artifacts,
        "progress_events": progress_events,
        "schema_version": site_obj.get("schema_version"),
    }

    json_path = run_dir / "prediction_log.json"
    md_path = run_dir / "prediction_report.md"
    html_path = run_dir / "prediction_report.html"
    json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(run_payload, report_view=report_view), encoding="utf-8")
    html_path.write_text(_render_html_report(run_payload, report_view=report_view), encoding="utf-8")
    orchestrator.record(
        plan="Finalize artifact bundle.",
        execution="write prediction_log.json, prediction_report.md, prediction_report.html",
        observation=f"artifacts={json_path.name},{md_path.name},{html_path.name}",
        status="ok",
    )
    _emit_progress(
        event_type="run_done",
        step_key="run_complete",
        label="Run completed",
        status="ok" if run_status == "ok" else "warn",
        details={"run_status": run_status, "run_id": run_id},
    )
    run_payload["orchestrator_trace"] = orchestrator.to_list()
    run_payload["progress_events"] = progress_events
    json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    orchestrator_file.write_text(json.dumps(orchestrator.to_list(), indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(run_payload, report_view=report_view), encoding="utf-8")
    html_path.write_text(_render_html_report(run_payload, report_view=report_view), encoding="utf-8")

    return {
        "run_payload": run_payload,
        "run_dir": run_dir,
        "json_path": json_path,
        "md_path": md_path,
        "html_path": html_path,
    }


def main() -> None:
    args = parse_args()

    uniprot = args.uniprot
    raw_sequence = args.sequence
    input_source = "sequence_arg"

    if args.interactive:
        uniprot, raw_sequence = interactive_input()
        input_source = "interactive"

    if args.sequence_file:
        file_sequence, file_header = read_sequence_file(args.sequence_file)
        raw_sequence = file_sequence
        input_source = f"sequence_file:{args.sequence_file}"
        if args.uniprot == "UNKNOWN" and file_header:
            uniprot = file_header.split()[0]

    if not raw_sequence:
        raw_sequence, resolved_source = resolve_sequence_from_uniprot(
            uniprot,
            allow_online_lookup=not args.no_online_lookup,
        )
        input_source = resolved_source

    result = run_prediction(
        uniprot=uniprot,
        raw_sequence=raw_sequence,
        llm_provider=args.llm_provider,
        checkpoint=None if args.use_base_model else args.checkpoint,
        base_model=args.base_model,
        openrouter_model=args.openrouter_model,
        openrouter_base_url=args.openrouter_base_url,
        openrouter_referer=args.openrouter_referer,
        openrouter_title=args.openrouter_title,
        openrouter_timeout=args.openrouter_timeout,
        mode=args.mode,
        candidate_source=args.candidate_source,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        self_consistency_k=args.self_consistency_k,
        sampling_seed=args.sampling_seed,
        max_input_tokens=args.max_input_tokens,
        candidate_pool_size=args.candidate_pool_size,
        chunk_size_aa=args.chunk_size_aa,
        chunk_overlap_aa=args.chunk_overlap_aa,
        output_dir=args.output_dir,
        require_api_key=True,
        input_source=input_source,
        enable_plot=not args.no_plot,
        repair_with_base_model=args.repair_with_base_model,
        panel_with_base_model=args.panel_with_base_model,
        ptm_source=args.ptm_source,
        ptm_policy=args.ptm_policy,
        motif_source=args.motif_source,
        use_motif=not args.no_motif,
        musitedeep_api_base_url=args.musitedeep_api_base_url,
        musitedeep_model_map=args.musitedeep_model_map,
        use_iedb_validation=not args.no_iedb_validation,
        iedb_table_path=args.iedb_table,
        iedb_iou_threshold=args.iedb_iou_threshold,
    )

    run_payload = result["run_payload"]
    print("=" * 80)
    print("Site4Drug Prediction Complete")
    print("=" * 80)
    print(f"Input label:          {uniprot}")
    print(f"Seq length:           {run_payload['input']['sequence_length']} aa")
    print(f"Requested mode:       {run_payload['input']['mode_request']}")
    print(f"Run status:           {run_payload.get('run_status', 'unknown')}")
    print(f"LLM provider:         {run_payload['model'].get('provider', 'tinker')}")
    print(f"Candidate source:     {run_payload['generation'].get('candidate_source')}")
    print(
        f"Recommended modality: {run_payload['recommended_modality']} "
        f"(conf={run_payload['modality_confidence']:.2f})"
    )
    print(f"Token strategy:       {run_payload['token_strategy_used']}")
    print(
        "Feature policy:       "
        f"ptm_source={run_payload['generation'].get('ptm_source')} | "
        f"ptm_policy={run_payload['generation'].get('ptm_policy')} | "
        f"motif_source={run_payload['generation'].get('motif_source')} | "
        f"use_motif={run_payload['generation'].get('use_motif')}"
    )
    iedb = run_payload.get("iedb_validation", {}) or {}
    print(
        "IEDB validation:      "
        f"enabled={iedb.get('enabled', False)} | "
        f"status={iedb.get('status', 'not_requested')} | "
        f"top_k_hit={iedb.get('top_k_hit', False)}"
    )
    print(f"Top-K req:            {run_payload['generation']['requested_top_k']}")
    print(f"JSON log:             {result['json_path']}")
    print(f"Markdown report:      {result['md_path']}")
    print(f"HTML report:          {result['html_path']}")


if __name__ == "__main__":
    main()
