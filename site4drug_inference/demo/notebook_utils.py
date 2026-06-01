#!/usr/bin/env python3
"""Notebook helper utilities for Site4Drug demos."""

from __future__ import annotations

import hashlib
import inspect
import importlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from site4drug_inference.common.env_utils import ensure_tinker_api_key
from site4drug_inference.demo import predict_site as _predict_site
from site4drug_inference.demo import ui_helpers as _ui_helpers
from site4drug_inference.demo.ui_helpers import infer_input_mode, ranked_candidates_display_df, resolve_demo_input_sequence

BASE_MODEL = _predict_site.BASE_MODEL
DEFAULT_BASE_MODEL = BASE_MODEL
DEFAULT_CHECKPOINT = _predict_site.DEFAULT_CHECKPOINT
DEFAULT_OUTPUT_DIR = _predict_site.DEFAULT_OUTPUT_DIR
DEFAULT_MUSITEDEEP_API_BASE_URL = getattr(_predict_site, "DEFAULT_MUSITEDEEP_API_BASE_URL", "https://www.musite.net")


def _load_predict_site():
    """Reload predict_site to avoid stale notebook kernel module state."""
    global _predict_site
    _predict_site = importlib.reload(_predict_site)
    _ui_helpers.predict_site = _predict_site
    return _predict_site


def ensure_api_key_or_raise(repo_root: Path) -> None:
    """Ensure TINKER_API_KEY is present in notebook sessions."""
    if ensure_tinker_api_key(repo_root):
        return
    raise RuntimeError(
        "TINKER_API_KEY is not set. Run ./scripts/setup_tinker_key.sh and source .tinker.env."
    )


def resolve_input_sequence(
    uniprot: str,
    sequence_text: str,
    sequence_file: str,
    allow_online_lookup: bool = True,
) -> tuple[str, str]:
    """Resolve sequence text/file/accession into normalized sequence + source label."""
    _load_predict_site()
    return resolve_demo_input_sequence(
        uniprot=uniprot,
        input_mode=infer_input_mode(sequence_text, sequence_file),
        sequence_text=sequence_text,
        sequence_file=sequence_file,
        allow_online_lookup=allow_online_lookup,
        paste_source_label="sequence_text",
        upload_source_prefix="sequence_file:",
    )


def run_notebook_prediction(
    *,
    uniprot: str,
    raw_sequence: str,
    checkpoint: str | None = DEFAULT_CHECKPOINT,
    base_model: str = DEFAULT_BASE_MODEL,
    use_base_model: bool = False,
    mode: str = "auto",
    candidate_source: str = "llm_propose",
    top_k: int = 5,
    max_tokens: int = 2048,
    self_consistency_k: int = 1,
    max_input_tokens: int = 10000,
    sampling_seed: int | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    enable_plot: bool = True,
    use_multi_agent: bool = True,
    input_source: str = "notebook",
    orchestrator_mode: str = "react",
    react_max_steps: int = 8,
    react_max_retries: int = 2,
    repair_with_base_model: bool = True,
    panel_with_base_model: bool = True,
    ptm_source: str = "musitedeep",
    ptm_policy: str = "tiered",
    motif_source: str = "remote",
    use_motif: bool = True,
    failure_policy: str = "raw_llm_only",
    musitedeep_api_base_url: str = DEFAULT_MUSITEDEEP_API_BASE_URL,
    musitedeep_model_map: str | Path | None = None,
    use_iedb_validation: bool = True,
    iedb_table_path: str | Path = REPO_ROOT / "data/tcell_regions_with_seq.parquet",
    iedb_iou_threshold: float = 0.3,
    show_progress: bool = True,
    progress_sink: Any | None = None,
) -> dict:
    """Notebook-friendly wrapper around predict_site.run_prediction."""
    predict_site = _load_predict_site()
    kwargs = dict(
        uniprot=uniprot,
        raw_sequence=raw_sequence,
        checkpoint=None if use_base_model else checkpoint,
        mode=mode,
        candidate_source=candidate_source,
        top_k=top_k,
        max_tokens=max_tokens,
        self_consistency_k=self_consistency_k,
        max_input_tokens=max_input_tokens,
        sampling_seed=sampling_seed,
        output_dir=output_dir,
        enable_plot=enable_plot,
        use_multi_agent=use_multi_agent,
        input_source=input_source,
        orchestrator_mode=orchestrator_mode,
        react_max_steps=react_max_steps,
        react_max_retries=react_max_retries,
        repair_with_base_model=repair_with_base_model,
        panel_with_base_model=panel_with_base_model,
        ptm_source=ptm_source,
        ptm_policy=ptm_policy,
        motif_source=motif_source,
        use_motif=use_motif,
        failure_policy=failure_policy,
        musitedeep_api_base_url=musitedeep_api_base_url,
        musitedeep_model_map=musitedeep_model_map,
        use_iedb_validation=use_iedb_validation,
        iedb_table_path=iedb_table_path,
        iedb_iou_threshold=iedb_iou_threshold,
    )
    if show_progress or progress_sink is not None:
        def _notebook_progress(event: dict[str, Any]) -> None:
            if progress_sink is not None:
                progress_sink(event)
            if show_progress:
                label = str(event.get("label", event.get("step_key", "step")))
                status = str(event.get("status", "running"))
                print(f"[{status}] {label}")
        kwargs["progress_callback"] = _notebook_progress
    # Backward-compatible call path if an older predict_site signature is loaded.
    sig = inspect.signature(predict_site.run_prediction)
    if "base_model" in sig.parameters:
        kwargs["base_model"] = base_model
    for key in list(kwargs.keys()):
        if key not in sig.parameters:
            kwargs.pop(key, None)
    return predict_site.run_prediction(**kwargs)


def set_reproducible_session(seed: int = 0) -> dict[str, Any]:
    """Apply deterministic notebook session knobs where possible."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except Exception:
        numpy_seeded = False
    return {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "python_version": sys.version.split()[0],
        "numpy_seeded": numpy_seeded,
    }


def _pkg_versions(names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not_installed"
    return out


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def topk_signature(run_payload: dict, top_n: int | None = None) -> list[str]:
    """Build a stable, compact signature for ranked candidates."""
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    rows = list(run_payload.get("ranked_candidates", []) or [])
    if top_n is not None:
        rows = rows[: int(top_n)]
    sig = []
    for row in rows:
        sig.append(
            "|".join(
                [
                    str(row.get("rank", "")),
                    str(row.get("candidate_id", "")),
                    f"{_safe_int(row.get('start', 0))}-{_safe_int(row.get('end', 0))}",
                    str(row.get("mode", "")),
                ]
            )
        )
    return sig

def validate_demo_run_payload(run_payload: dict) -> dict[str, Any]:
    """Run lightweight schema/consistency checks for notebook demos."""
    issues: list[str] = []
    ranked = run_payload.get("ranked_candidates", []) or []
    if not ranked:
        issues.append("ranked_candidates_empty")
    for idx, row in enumerate(ranked, start=1):
        for key in ("candidate_id", "start", "end", "peptide"):
            if key not in row:
                issues.append(f"candidate_{idx}_missing_{key}")
        start = int(row.get("start", 0) or 0)
        end = int(row.get("end", 0) or 0)
        pep = str(row.get("peptide", "") or "")
        if start < 1 or end < start:
            issues.append(f"candidate_{idx}_invalid_span:{start}-{end}")
        if pep and (end - start + 1) != len(pep):
            issues.append(f"candidate_{idx}_span_length_mismatch")
    return {"ok": not issues, "issues": issues}

def build_repro_manifest(
    *,
    result: dict,
    config: dict[str, Any],
    session_info: dict[str, Any] | None = None,
    notebook_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a machine-readable reproducibility manifest for one notebook run."""
    run_payload = result.get("run_payload", {}) or {}
    run_dir = Path(result.get("run_dir", REPO_ROOT / "outputs/predictions"))

    artifact_paths: list[Path] = []
    for key in ("json_path", "md_path", "html_path"):
        value = result.get(key)
        if value:
            artifact_paths.append(Path(value))
    plot_artifacts = run_payload.get("plot_artifacts", {}) or {}
    plot_png = plot_artifacts.get("plot_png_name")
    plot_json = plot_artifacts.get("plot_json_name")
    if plot_png:
        artifact_paths.append(run_dir / str(plot_png))
    if plot_json:
        artifact_paths.append(run_dir / str(plot_json))
    agent_traces = run_payload.get("agent_artifacts", {}) or {}
    for rel in agent_traces.values():
        artifact_paths.append(run_dir / str(rel))
    artifact_paths.append(run_dir / "orchestrator_trace.json")

    seen = set()
    artifact_rows = []
    for path in artifact_paths:
        p = Path(path)
        if p in seen:
            continue
        seen.add(p)
        if not p.exists():
            continue
        artifact_rows.append(
            {
                "path": str(p),
                "relative_path": str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p),
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            }
        )

    input_obj = run_payload.get("input", {}) or {}
    seq = str(input_obj.get("sequence", "") or "")
    manifest = {
        "manifest_version": "site4drug_notebook_repro_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "notebook_path": str(notebook_path) if notebook_path else None,
        "run_id": run_payload.get("run_id"),
        "run_dir": str(run_dir),
        "session": session_info or {},
        "platform": {
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _pkg_versions(["tinker", "tinker-cookbook", "numpy", "pandas", "matplotlib", "pyarrow"]),
        },
        "config": config,
        "input_fingerprint": {
            "uniprot": input_obj.get("uniprot"),
            "sequence_length": int(input_obj.get("sequence_length", 0) or 0),
            "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
            "source": input_obj.get("source"),
            "mode_request": input_obj.get("mode_request"),
        },
        "runtime_outcome": {
            "recommended_modality": run_payload.get("recommended_modality"),
            "modality_confidence": run_payload.get("modality_confidence"),
            "token_strategy_used": run_payload.get("token_strategy_used"),
            "parser_status": (run_payload.get("parser_meta", {}) or {}).get("parser_status"),
            "topk_signature": topk_signature(run_payload),
            "audit_warnings": (run_payload.get("audit_log", {}) or {}).get("warnings", []),
        },
        "schema_check": validate_demo_run_payload(run_payload),
        "artifacts": artifact_rows,
        "notes": [
            "Temperature=0 and fixed config improve repeatability, but hosted model/runtime changes can still alter outputs.",
            "Use artifact hashes + topk signature to compare reruns.",
        ],
    }
    return manifest


def save_repro_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a reproducibility manifest and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def ranked_candidates_df(run_payload: dict) -> pd.DataFrame:
    """Convert ranked candidates to DataFrame for display."""
    rows = list(run_payload.get("ranked_candidates", []) or [])
    if not rows:
        return pd.DataFrame(columns=["rank", "candidate_id", "mode", "peptide", "start", "end", "confidence", "confidence_score", "confidence_source", "flags", "reason"])
    df = pd.DataFrame(rows)
    cols = [
        "rank",
        "candidate_id",
        "mode",
        "peptide",
        "start",
        "end",
        "confidence",
        "confidence_score",
        "confidence_source",
        "flags",
        "reason",
    ]
    return df[[c for c in cols if c in df.columns]]


def ranked_candidates_ui_df(run_payload: dict) -> pd.DataFrame:
    """Compact ranked-candidates table matching the interactive demo surfaces."""
    return ranked_candidates_display_df(run_payload)


def candidate_evidence_df(run_payload: dict) -> pd.DataFrame:
    """Convert candidate evidence to DataFrame for display."""
    return pd.DataFrame(run_payload.get("candidate_evidence", []))


def raw_api_calls_df(run_payload: dict) -> pd.DataFrame:
    """Tabular view of raw API call metadata."""
    raw = run_payload.get("raw_api_calls", {}) or {}
    mus = raw.get("musitedeep", {}) if isinstance(raw, dict) else {}
    pro = raw.get("scanprosite", {}) if isinstance(raw, dict) else {}
    rows = [
        {
            "api": "musitedeep",
            "status": mus.get("status", ""),
            "artifact_path": mus.get("artifact_path", ""),
            "request_count": mus.get("request_count", 0),
            "n_hits": None,
            "preview": mus.get("preview", ""),
            "error_or_warning": mus.get("errors", ""),
        },
        {
            "api": "scanprosite",
            "status": pro.get("status", ""),
            "artifact_path": pro.get("artifact_path", ""),
            "request_count": None,
            "n_hits": pro.get("n_hits", 0),
            "preview": pro.get("preview", ""),
            "error_or_warning": pro.get("error", "") or pro.get("warning", ""),
        },
    ]
    return pd.DataFrame(rows)


def save_run_snapshot(run_payload: dict, output_path: str | Path) -> None:
    """Persist run payload snapshot for notebook experimentation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
