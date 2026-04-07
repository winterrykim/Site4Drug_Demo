#!/usr/bin/env python3
"""LLM-based multi-agent reasoning helpers for Site4Drug."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from site4drug_inference.common.tinker_sampling import build_sampling_params
from site4drug_inference.demo.output_parser import parse_json_object
from site4drug_inference.demo.prompt_templates import (
    AGENT_SYSTEM_PROMPTS,
    build_decision_prompt,
    build_specialist_prompt,
)

SPECIALIST_REQUIRED_KEYS = (
    "agent",
    "modality_votes",
    "candidate_adjustments",
    "risk_flags",
    "summary",
)

DECISION_REQUIRED_KEYS = (
    "recommended_modality",
    "modality_confidence",
    "ranking",
    "global_risks",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _contains_any(text: str, keywords: set[str]) -> bool:
    low = str(text).lower()
    return any(k in low for k in keywords)


def _rank_candidates(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda c: _safe_float(c.get("heuristic_score"), 0.0), reverse=True)


def _normalize_votes(raw_votes: dict[str, float]) -> dict[str, float]:
    cleaned = {
        "epitope": max(_safe_float(raw_votes.get("epitope"), 0.0), 0.0),
        "pocket": max(_safe_float(raw_votes.get("pocket"), 0.0), 0.0),
        "other": max(_safe_float(raw_votes.get("other"), 0.0), 0.0),
    }
    total = sum(cleaned.values())
    if total <= 0:
        return {"epitope": 0.5, "pocket": 0.3, "other": 0.2}
    return {k: v / total for k, v in cleaned.items()}


def _sample_json(
    sampling_client,
    renderer,
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1100,
    sampling_seed: int | None = None,
) -> tuple[dict | None, dict]:
    from tinker import types

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = renderer.build_generation_prompt(messages)
    params, _ = build_sampling_params(
        types,
        max_tokens=max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
        sampling_seed=sampling_seed,
    )
    raw_text = tokenizer.decode(
        sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        .result()
        .sequences[0]
        .tokens
    )
    payload, err = parse_json_object(raw_text)
    return payload, {"raw_text": raw_text, "parse_error": err}


def _repair_json_once(
    sampling_client,
    renderer,
    tokenizer,
    *,
    system_prompt: str,
    original_user_prompt: str,
    raw_text: str,
    repair_prompt: str,
    max_tokens: int,
    sampling_seed: int | None = None,
) -> tuple[dict | None, dict]:
    from tinker import types

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_user_prompt},
        {"role": "assistant", "content": raw_text},
        {"role": "user", "content": repair_prompt},
    ]
    prompt = renderer.build_generation_prompt(messages)
    params, _ = build_sampling_params(
        types,
        max_tokens=max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
        sampling_seed=sampling_seed,
    )
    repaired_text = tokenizer.decode(
        sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        .result()
        .sequences[0]
        .tokens
    )
    payload, err = parse_json_object(repaired_text)
    return payload, {"repair_raw_text": repaired_text, "repair_parse_error": err}


def _validate_specialist_payload(
    payload: dict | None,
    expected_agent: str,
    has_ptm: bool = False,
    has_motif: bool = False,
) -> list[str]:
    errors: list[str] = []
    if payload is None:
        return ["payload_missing"]
    if not isinstance(payload, dict):
        return ["payload_not_object"]
    for key in SPECIALIST_REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"missing_{key}")
    if payload.get("agent") != expected_agent:
        errors.append("agent_name_mismatch")
    if not isinstance(payload.get("modality_votes", {}), dict):
        errors.append("modality_votes_not_object")
    if not isinstance(payload.get("risk_flags", []), list):
        errors.append("risk_flags_not_list")
    if not isinstance(payload.get("candidate_adjustments", []), list):
        errors.append("candidate_adjustments_not_list")
    if not isinstance(payload.get("summary", ""), str):
        errors.append("summary_not_string")
    combined_text = str(payload.get("summary", ""))
    for row in payload.get("candidate_adjustments", []):
        if isinstance(row, dict):
            combined_text += " " + str(row.get("reason", ""))
            evidence = row.get("evidence", [])
            if isinstance(evidence, list):
                combined_text += " " + " ".join(str(item) for item in evidence)
            elif isinstance(evidence, dict):
                combined_text += " " + " ".join(f"{k}:{v}" for k, v in evidence.items())
    if has_ptm and not _contains_any(
        combined_text,
        {"ptm", "glyco", "glycosyl", "phospho", "ubiquit", "acetyl", "methyl", "hydroxy"},
    ):
        errors.append("missing_ptm_reasoning")
    if has_motif and not _contains_any(combined_text, {"motif", "zinc", "nls", "dna", "helix", "zipper"}):
        errors.append("missing_motif_reasoning")
    return errors


def _validate_decision_payload(payload: dict | None, has_ptm: bool = False, has_motif: bool = False) -> list[str]:
    errors: list[str] = []
    if payload is None:
        return ["payload_missing"]
    if not isinstance(payload, dict):
        return ["payload_not_object"]
    for key in DECISION_REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"missing_{key}")
    if payload.get("recommended_modality") not in {"epitope", "pocket", "other"}:
        errors.append("invalid_recommended_modality")
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        errors.append("ranking_not_list")
    else:
        for i, row in enumerate(ranking):
            if not isinstance(row, dict):
                errors.append(f"ranking_row_{i}_not_object")
                continue
            for key in ("rank", "candidate_id", "reason"):
                if key not in row:
                    errors.append(f"ranking_row_{i}_missing_{key}")
            if "confidence_score" in row:
                score = _safe_float(row.get("confidence_score"), -1.0)
                if score < 0.0 or score > 1.0:
                    errors.append(f"ranking_row_{i}_confidence_score_out_of_range")
    if not isinstance(payload.get("global_risks", []), list):
        errors.append("global_risks_not_list")
    ranking_blob = " ".join(str(r.get("reason", "")) for r in payload.get("ranking", []) if isinstance(r, dict))
    confidence_blob = " ".join(
        str(r.get("confidence_reason", "")) for r in payload.get("ranking", []) if isinstance(r, dict)
    )
    risks_blob = " ".join(str(x) for x in payload.get("global_risks", []) if isinstance(x, str))
    combined = f"{ranking_blob} {confidence_blob} {risks_blob}"
    if has_ptm and not _contains_any(
        combined,
        {"ptm", "glyco", "glycosyl", "phospho", "ubiquit", "acetyl", "methyl", "hydroxy"},
    ):
        errors.append("missing_ptm_reasoning")
    if has_motif and not _contains_any(combined, {"motif", "zinc", "nls", "dna", "helix", "zipper"}):
        errors.append("missing_motif_reasoning")
    return errors


def _top_candidates_for_prompt(candidates: list[dict], top_n: int = 25) -> list[dict]:
    limited = []
    for c in candidates[:top_n]:
        motif_hits = c.get("motif_hits_overlapping", [])
        if isinstance(motif_hits, list):
            motif_hits = motif_hits[:3]
        else:
            motif_hits = []
        limited.append(
            {
                "candidate_id": c.get("candidate_id"),
                "mode": c.get("mode"),
                "start": c.get("start"),
                "end": c.get("end"),
                "peptide": c.get("peptide"),
                "mean_hydropathy": round(_safe_float(c.get("mean_hydropathy"), 0.0), 3),
                "hydrophobic_fraction": round(_safe_float(c.get("hydrophobic_fraction"), 0.0), 3),
                "polar_fraction": round(_safe_float(c.get("polar_fraction"), 0.0), 3),
                "overlaps_tm": bool(c.get("overlaps_tm")),
                "overlaps_ptm_mask": bool(c.get("overlaps_ptm_mask")),
                "ptm_overlap_by_type": c.get("ptm_overlap_by_type", {}),
                "ptm_density": round(_safe_float(c.get("ptm_density"), 0.0), 4),
                "motif_hit_count": int(_safe_float(c.get("motif_hit_count"), 0.0)),
                "motif_hits_overlapping": motif_hits,
                "risk_flags": c.get("risk_flags", []),
                "heuristic_score": round(_safe_float(c.get("heuristic_score"), 0.0), 4),
            }
        )
    return limited


def _specialist_votes(agent_name: str, ranked: list[dict]) -> dict[str, float]:
    if not ranked:
        return {"epitope": 0.5, "pocket": 0.3, "other": 0.2}
    top = ranked[: min(len(ranked), 12)]
    epi_pref = 0.0
    pocket_pref = 0.0
    risk_mass = 0.0
    for c in top:
        score = _safe_float(c.get("heuristic_score"), 0.0)
        epi_bonus = 0.3 if c.get("mode") == "epitope" else 0.0
        pocket_bonus = 0.3 if c.get("mode") == "pocket" else 0.0
        no_tm = 1.0 if not c.get("overlaps_tm") else 0.0
        no_ptm = 1.0 if not c.get("overlaps_ptm_mask") else 0.0
        hydrophobic = _safe_float(c.get("hydrophobic_fraction"), 0.0)
        polar = _safe_float(c.get("polar_fraction"), 0.0)
        flag_count = len(c.get("risk_flags", []))

        epi_pref += no_tm + no_ptm + polar + epi_bonus + max(score, 0.0) * 0.15
        pocket_pref += hydrophobic + pocket_bonus + max(score, 0.0) * 0.25
        risk_mass += flag_count + (1.0 if c.get("overlaps_tm") else 0.0) + (1.0 if c.get("overlaps_ptm_mask") else 0.0)

    if agent_name == "BioAgent":
        raw = {"epitope": epi_pref * 1.25, "pocket": pocket_pref * 0.85, "other": risk_mass * 0.20}
    elif agent_name == "ChemAgent":
        raw = {"epitope": epi_pref * 0.85, "pocket": pocket_pref * 1.25, "other": risk_mass * 0.15}
    else:  # RiskAgent
        raw = {"epitope": epi_pref * 0.70, "pocket": pocket_pref * 0.70, "other": risk_mass * 0.80 + 0.25}
    return _normalize_votes(raw)


def _fallback_specialist_payload(
    agent_name: str,
    candidates: list[dict],
    sequence_summary: dict[str, Any],
) -> dict:
    ranked = _rank_candidates(candidates)
    votes = _specialist_votes(agent_name, ranked)
    top = ranked[: min(6, len(ranked))]

    risk_flags: set[str] = set()
    adjustments = []
    for c in top:
        cid = c.get("candidate_id")
        mode = c.get("mode", "other")
        hydrophobic = _safe_float(c.get("hydrophobic_fraction"), 0.0)
        polar = _safe_float(c.get("polar_fraction"), 0.0)
        overlaps_tm = bool(c.get("overlaps_tm"))
        overlaps_ptm = bool(c.get("overlaps_ptm_mask"))
        flags = [str(f) for f in c.get("risk_flags", [])]
        risk_flags.update(flags)

        if agent_name == "BioAgent":
            delta = (0.18 if not overlaps_tm else -0.25) + (0.12 if not overlaps_ptm else -0.18)
            reason = (
                "Biological accessibility preference: favors non-TM, non-PTM-overlapping regions."
                if delta >= 0
                else "Biological constraint penalty from TM/PTM overlap."
            )
        elif agent_name == "ChemAgent":
            delta = (0.16 if mode == "pocket" and hydrophobic >= 0.45 else 0.06 if polar >= 0.30 else -0.06)
            reason = (
                "Chemical surface compatibility: hydrophobic signature supports pocket-oriented binding."
                if delta >= 0
                else "Chemical plausibility weaker for this composition profile."
            )
        else:
            penalty = 0.08 * len(flags) + (0.10 if overlaps_tm else 0.0) + (0.10 if overlaps_ptm else 0.0)
            delta = -penalty
            reason = "Risk penalty from accumulated constraint flags."

        adjustments.append(
            {
                "candidate_id": cid,
                "delta": round(delta, 4),
                "reason": reason,
                "evidence": [
                    f"mode={mode}",
                    f"overlaps_tm={overlaps_tm}",
                    f"overlaps_ptm_mask={overlaps_ptm}",
                    f"hydrophobic_fraction={round(hydrophobic, 3)}",
                    f"polar_fraction={round(polar, 3)}",
                    f"risk_flags={flags}",
                ],
            }
        )

    if not risk_flags:
        risk_flags.add("no_major_constraint_violation_detected")

    summary = {
        "BioAgent": (
            "Topology/PTM screen prioritizes biologically accessible regions and downranks constrained segments."
        ),
        "ChemAgent": (
            "Hydropathy and residue composition suggest chemistry-aligned candidates with mode-sensitive feasibility."
        ),
        "RiskAgent": (
            "Risk review penalizes candidates with stacked flags and highlights uncertainty hotspots."
        ),
    }[agent_name]

    return {
        "agent": agent_name,
        "modality_votes": votes,
        "candidate_adjustments": adjustments[:8],
        "risk_flags": sorted(risk_flags),
        "summary": summary,
        "fallback_generated": True,
        "context_snapshot": {
            "sequence_length": sequence_summary.get("sequence_length"),
            "n_tm_regions": len(sequence_summary.get("tm_regions", [])),
            "n_ptm_sites": len(sequence_summary.get("ptm_sites", [])),
            "n_cysteines": len(sequence_summary.get("cysteine_positions", [])),
        },
    }


def _aggregate_adjustments(candidates: list[dict], specialists: list[dict]) -> dict[str, float]:
    scores = {str(c.get("candidate_id")): _safe_float(c.get("heuristic_score"), 0.0) for c in candidates}
    for specialist in specialists:
        for row in specialist.get("candidate_adjustments", []):
            cid = str(row.get("candidate_id"))
            if cid not in scores:
                continue
            scores[cid] += _safe_float(row.get("delta"), 0.0)
    return scores


def _deterministic_decision(
    candidates: list[dict],
    specialists: list[dict],
    forced_mode: str | None = None,
    top_k: int = 5,
) -> dict:
    votes = {"epitope": 0.0, "pocket": 0.0, "other": 0.0}
    for specialist in specialists:
        sv = _normalize_votes(specialist.get("modality_votes", {}))
        for key in votes:
            votes[key] += sv.get(key, 0.0)
    votes = _normalize_votes(votes)

    mode = forced_mode if forced_mode in {"epitope", "pocket"} else max(votes, key=votes.get)
    candidate_scores = _aggregate_adjustments(candidates, specialists)
    ordered = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
    by_id = {str(c.get("candidate_id")): c for c in candidates}

    ranking = []
    for cid, score in ordered:
        c = by_id.get(cid)
        if not c:
            continue
        if mode in {"epitope", "pocket"} and c.get("mode") != mode:
            continue
        ranking.append(
            {
                "rank": len(ranking) + 1,
                "candidate_id": cid,
                "reason": f"Consensus-adjusted heuristic score {score:.3f}",
                "confidence_score": 0.58,
                "confidence_reason": "Fallback estimate from consensus-adjusted score.",
            }
        )
        if len(ranking) >= top_k:
            break
    if not ranking:
        for cid, score in ordered[:top_k]:
            ranking.append(
                {
                    "rank": len(ranking) + 1,
                    "candidate_id": cid,
                    "reason": f"Fallback score {score:.3f}",
                    "confidence_score": 0.50,
                    "confidence_reason": "Fallback estimate from heuristic-only ranking.",
                }
            )

    global_risks: set[str] = {"multi_agent_fallback_used"}
    for specialist in specialists:
        global_risks.update(str(flag) for flag in specialist.get("risk_flags", []))

    return {
        "recommended_modality": mode,
        "modality_confidence": round(votes.get(mode, 0.5), 3),
        "ranking": ranking,
        "global_risks": sorted(global_risks),
    }


def _decision_repair_prompt(
    top_k: int,
    errors: list[str],
    has_ptm: bool,
    has_motif: bool,
    forced_mode: str = "",
) -> str:
    notes: list[str] = []
    if forced_mode in {"epitope", "pocket"}:
        notes.append(
            f"Requested modality is fixed to {forced_mode}. "
            "Set recommended_modality to this value and keep ranking within that modality."
        )
    if has_ptm:
        notes.append("PTM context exists, so ranking reasons must briefly mention PTM evidence or caveats.")
    if has_motif:
        notes.append("Motif context exists, so ranking reasons must briefly mention motif evidence or caveats.")
    notes_block = "\n".join(notes)
    notes_prefix = f"{notes_block}\n" if notes_block else ""
    return (
        "Your previous response failed validation.\n"
        "Return ONLY one valid JSON object with keys: "
        "recommended_modality, modality_confidence, ranking, global_risks.\n"
        "Do not include markdown, comments, or any text before/after the JSON object.\n"
        f"ranking must contain at most {int(top_k)} rows of "
        "{rank, candidate_id, reason, confidence_score, confidence_reason}.\n"
        f"Current issues to fix: {errors}\n"
        f"{notes_prefix}"
        '{\n'
        '  "recommended_modality": "epitope|pocket|other",\n'
        '  "modality_confidence": 0.0,\n'
        '  "ranking": [{"rank": 1, "candidate_id": "C0001", "reason": "Short evidence-grounded rationale.", '
        '"confidence_score": 0.0, "confidence_reason": "Short confidence note."}],\n'
        '  "global_risks": ["..."]\n'
        '}'
    )


def _specialist_prompt(agent_name: str, context_json: str) -> str:
    return build_specialist_prompt(agent_name, context_json)


def _decision_prompt(context_json: str, top_k: int, requested_mode: str = "auto") -> str:
    return build_decision_prompt(context_json, top_k, requested_mode=requested_mode)


def run_multi_agent_reasoning(
    sampling_client,
    renderer,
    tokenizer,
    sequence_summary: dict[str, Any],
    candidates: list[dict],
    requested_mode: str = "auto",
    top_k: int = 5,
    deterministic_only: bool = False,
    progress_callback=None,
    sampling_seed: int | None = None,
) -> tuple[dict, dict]:
    """Run Bio/Chem/Risk/Decision panel and return decision + trace metadata."""
    if deterministic_only:
        deterministic_only = False

    def _emit_progress(
        *,
        event_type: str,
        step_key: str,
        label: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if progress_callback is None:
            return
        payload = {
            "event_type": event_type,
            "step_key": step_key,
            "label": label,
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if details is not None:
            payload["details"] = details
        try:
            progress_callback(payload)
        except Exception:
            pass

    prompt_candidates = _top_candidates_for_prompt(candidates, top_n=min(20, max(10, top_k + 5)))
    ptm_sites_full = sequence_summary.get("ptm_sites", []) or []
    motif_hits_full = sequence_summary.get("motif_hits", []) or []
    compact_context = {
        "sequence_length": sequence_summary.get("sequence_length"),
        "tm_regions": sequence_summary.get("tm_regions", [])[:20],
        "ptm_site_count": len(ptm_sites_full),
        "ptm_sites_sample": ptm_sites_full[:30],
        "ptm_summary": sequence_summary.get("ptm_summary", {}),
        "motif_hit_count": len(motif_hits_full),
        "motif_hits_sample": motif_hits_full[:20],
        "motif_summary": sequence_summary.get("motif_summary", {}),
        "n_cysteines": len(sequence_summary.get("cysteine_positions", [])),
        "requested_mode": requested_mode,
        "top_k": top_k,
    }
    base_context_json = json.dumps({"context": compact_context, "candidates": prompt_candidates}, ensure_ascii=False)

    has_ptm = bool(ptm_sites_full)
    has_motif = bool(motif_hits_full)
    forced = requested_mode if requested_mode in {"epitope", "pocket"} else None
    specialist_max_tokens = min(700, max(500, 420 + top_k * 20))
    decision_max_tokens = min(900, max(700, 520 + top_k * 40))

    specialist_outputs: dict[str, dict] = {}
    specialist_traces: dict[str, dict] = {}
    specialist_errors: dict[str, list[str]] = {}

    for idx, agent_name in enumerate(("BioAgent", "ChemAgent", "RiskAgent"), start=1):
        _emit_progress(
            event_type="agent_start",
            step_key=agent_name,
            label=agent_name,
            status="running",
        )
        payload, trace = _sample_json(
            sampling_client,
            renderer,
            tokenizer,
            system_prompt=AGENT_SYSTEM_PROMPTS[agent_name],
            user_prompt=_specialist_prompt(agent_name, base_context_json),
            max_tokens=specialist_max_tokens,
            sampling_seed=(int(sampling_seed) + idx if sampling_seed is not None else None),
        )
        errors = _validate_specialist_payload(
            payload,
            agent_name,
            has_ptm=has_ptm,
            has_motif=has_motif,
        )
        if errors:
            payload = _fallback_specialist_payload(agent_name, candidates, sequence_summary)
        specialist_outputs[agent_name] = payload
        specialist_traces[f"{agent_name.lower().replace('agent', '_agent')}"] = trace
        specialist_errors[f"{agent_name.lower().replace('agent', '_agent')}"] = errors
        _emit_progress(
            event_type="agent_done",
            step_key=agent_name,
            label=agent_name,
            status="warn" if errors else "ok",
            details={
                "fallback_used": bool(errors),
                "validation_errors": list(errors),
            },
        )

    decision_context = json.dumps(
        {
            "context": compact_context,
            "bio": specialist_outputs["BioAgent"],
            "chem": specialist_outputs["ChemAgent"],
            "risk": specialist_outputs["RiskAgent"],
            "candidates": prompt_candidates,
        },
        ensure_ascii=False,
    )
    _emit_progress(
        event_type="agent_start",
        step_key="DecisionAgent",
        label="DecisionAgent",
        status="running",
    )
    decision_payload, decision_trace = _sample_json(
        sampling_client,
        renderer,
        tokenizer,
        system_prompt=AGENT_SYSTEM_PROMPTS["DecisionAgent"],
        user_prompt=_decision_prompt(decision_context, top_k=top_k, requested_mode=forced or "auto"),
        max_tokens=decision_max_tokens,
        sampling_seed=(int(sampling_seed) + 100 if sampling_seed is not None else None),
    )
    decision_errors = _validate_decision_payload(
        decision_payload,
        has_ptm=has_ptm,
        has_motif=has_motif,
    )

    decision_repair_attempted = False
    if decision_errors:
        decision_repair_attempted = True
        repaired_payload, repair_trace = _repair_json_once(
            sampling_client,
            renderer,
            tokenizer,
            system_prompt=AGENT_SYSTEM_PROMPTS["DecisionAgent"],
            original_user_prompt=_decision_prompt(decision_context, top_k=top_k, requested_mode=forced or "auto"),
            raw_text=str(decision_trace.get("raw_text", "")),
            repair_prompt=_decision_repair_prompt(
                top_k,
                decision_errors,
                has_ptm,
                has_motif,
                forced_mode=forced or "",
            ),
            max_tokens=decision_max_tokens,
            sampling_seed=(int(sampling_seed) + 101 if sampling_seed is not None else None),
        )
        decision_trace.update(repair_trace)
        if repaired_payload is not None:
            repaired_errors = _validate_decision_payload(
                repaired_payload,
                has_ptm=has_ptm,
                has_motif=has_motif,
            )
            if not repaired_errors:
                decision_payload = repaired_payload
                decision_errors = []
            else:
                decision_errors = repaired_errors

    decision_fallback_used = False
    if decision_errors:
        seed = _deterministic_decision(
            candidates=candidates,
            specialists=list(specialist_outputs.values()),
            forced_mode=forced,
            top_k=top_k,
        )
        decision_payload = {
            "recommended_modality": seed.get("recommended_modality", forced or "epitope"),
            "modality_confidence": _safe_float(seed.get("modality_confidence"), 0.0),
            "ranking": [],
            "global_risks": sorted(
                {
                    *[str(flag) for flag in seed.get("global_risks", [])],
                    "decision_output_invalid",
                }
            ),
        }
    else:
        if forced:
            decision_payload["recommended_modality"] = forced
        ranking = decision_payload.get("ranking", [])
        by_id = {c.get("candidate_id"): c for c in candidates}
        filtered = [r for r in ranking if r.get("candidate_id") in by_id]
        filtered.sort(key=lambda x: x.get("rank", 1))
        base_modality_conf = min(
            max(_safe_float(decision_payload.get("modality_confidence"), 0.5), 0.0),
            1.0,
        )
        normalized_rows = []
        for row in filtered:
            out = dict(row)
            rank_idx = max(int(_safe_float(out.get("rank"), 1)), 1)
            default_score = max(min(base_modality_conf - 0.04 * (rank_idx - 1), 0.95), 0.35)
            out["confidence_score"] = min(max(_safe_float(out.get("confidence_score"), default_score), 0.0), 1.0)
            out.setdefault("confidence_reason", "Decision-agent self-estimate from evidence consistency.")
            normalized_rows.append(out)
        decision_payload["ranking"] = normalized_rows[:top_k]
        decision_payload["modality_confidence"] = round(base_modality_conf, 3)
        if decision_payload.get("recommended_modality") not in {"epitope", "pocket", "other"}:
            decision_payload["recommended_modality"] = forced or "epitope"

    panel_status = "ok"
    if decision_errors:
        panel_status = "decision_invalid"
    elif not decision_payload.get("ranking"):
        panel_status = "decision_empty"

    traces = {
        "bio_agent": {
            "parsed": specialist_outputs["BioAgent"],
            "validation_errors": specialist_errors["bio_agent"],
            "fallback_used": bool(specialist_errors["bio_agent"]),
            **specialist_traces["bio_agent"],
        },
        "chem_agent": {
            "parsed": specialist_outputs["ChemAgent"],
            "validation_errors": specialist_errors["chem_agent"],
            "fallback_used": bool(specialist_errors["chem_agent"]),
            **specialist_traces["chem_agent"],
        },
        "risk_agent": {
            "parsed": specialist_outputs["RiskAgent"],
            "validation_errors": specialist_errors["risk_agent"],
            "fallback_used": bool(specialist_errors["risk_agent"]),
            **specialist_traces["risk_agent"],
        },
        "decision_agent": {
            "parsed": decision_payload,
            "validation_errors": decision_errors,
            "fallback": decision_fallback_used,
            "repair_attempted": decision_repair_attempted,
            **decision_trace,
        },
        "panel_status": panel_status,
    }
    _emit_progress(
        event_type="agent_done",
        step_key="DecisionAgent",
        label="DecisionAgent",
        status="warn" if decision_errors or decision_fallback_used else "ok",
        details={
            "fallback_used": bool(decision_fallback_used),
            "validation_errors": list(decision_errors),
            "repair_attempted": bool(decision_repair_attempted),
            "panel_status": panel_status,
        },
    )
    return decision_payload, traces
