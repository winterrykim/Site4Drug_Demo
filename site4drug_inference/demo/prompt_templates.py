#!/usr/bin/env python3
"""Central prompt templates for Site4Drug inference and multi-agent steps.

Edit this file to adjust prompt wording without changing pipeline logic.
"""

from __future__ import annotations

from typing import Any

SITE_SYSTEM_PROMPT = (
    "You are Site4Drug, an expert system for constraint-first targetable site discovery on proteins. "
    "You must decide target modality (epitope vs pocket) unless explicitly fixed. "
    "Epitope mode predicts where antibodies or peptide binders attach, usually on extracellular protruding regions of membrane proteins, often tied to immune responses. Pocket mode predicts where small molecules bind, typically in intracellular proteins or membrane channels, covering immune as well as general signaling and metabolic pathways. "
    "Follow a constraint hierarchy: topology/TM -> PTM masks -> motif-functional caveats -> disulfide/context. "
    "Generate or rank targetable candidates with evidence-grounded rationale. Always output strict JSON."
)


def mode_policy_text(mode: str) -> str:
    if mode != "auto":
        return f"Use forced modality: {mode}. Set recommended_modality='{mode}' and keep candidate modes consistent with this setting."
    return "Choose modality automatically using the provided evidence only. Keep each candidate mode consistent with your final recommended_modality."


def schema_instruction(candidate_source: str, top_k: int) -> str:
    _ = candidate_source
    return (
        "Return ONLY JSON with keys:\n"
        "{\n"
        '  "recommended_modality": "epitope|pocket|other",\n'
        '  "modality_confidence": 0.0,\n'
        '  "ranked_candidates": [\n'
        '    {"rank": 1, "start": 1, "end": 15, "peptide": "AAAA", "mode": "epitope|pocket|other", "confidence_score": 0.0, "reason": "..."}\n'
        "  ]\n"
        "}\n"
        f"ranked_candidates should contain up to Top-{top_k} rows.\n"
        "If recommended_modality is epitope or pocket, prefer ranked_candidates rows in the same mode unless evidence strongly supports a mixed set.\n"
        "Each ranked row must stay span-consistent (peptide length should match end-start+1).\n"
        "When PTM or motif signals exist, briefly mention PTM/motif caveats in the reason.\n"
        "candidate_id is optional and will be assigned later.\n"
        "Output must start with '{' and end with '}'. Do not output markdown, XML tags, or extra prose."
    )


def _auto_policy_block(auto_mode_policy: dict[str, Any] | None) -> str:
    if not isinstance(auto_mode_policy, dict):
        return ""
    return (
        "Auto mode policy (deterministic project rule):\n"
        "- Non-membrane protein -> pocket\n"
        "- Membrane protein -> pocket or epitope\n"
        f"- This target policy decision: {auto_mode_policy.get('mode')} "
        f"(n_tm_regions={int(float(auto_mode_policy.get('n_tm_regions', 0) or 0))}, "
        f"channel_like={bool(auto_mode_policy.get('channel_like', False))})\n"
        f"- Policy reason: {auto_mode_policy.get('reason', '')}\n"
        "Set recommended_modality to this policy decision.\n\n"
        "When emitting ranked_candidates, prefer row modes that match this policy decision.\n\n"
    )


def build_full_sequence_user_prompt(
    *,
    uniprot: str,
    sequence: str,
    mode: str,
    top_k: int,
    seq_summary: dict,
    candidate_source: str,
    auto_mode_policy: dict[str, Any] | None,
    ptm_text: str,
    motif_text: str,
) -> str:
    mode_instruction = mode_policy_text(mode)
    auto_block = _auto_policy_block(auto_mode_policy if mode == "auto" else None)
    return (
        "Objective:\n"
        "Identify targetable site regions using constraint-first analysis and return strict JSON.\n\n"
        "Modality policy:\n"
        f"{mode_instruction}\n\n"
        f"{auto_block}"
        "Constraint hierarchy:\n"
        "1) TM/topology constraints\n"
        "2) PTM mask constraints (typed)\n"
        "3) Motif-functional caveats\n"
        "4) Disulfide/context risks\n\n"
        f"Target: UniProt {uniprot}\n"
        f"Length: {len(sequence)} aa\n"
        f"TM regions: {seq_summary.get('tm_regions', [])}\n"
        f"Cysteines: {len(seq_summary.get('cysteine_positions', []))}\n\n"
        f"{ptm_text}\n\n"
        f"{motif_text}\n\n"
        f"Antigen sequence:\n{sequence}\n\n"
        f"Output request:\nReturn Top-{top_k} candidates.\n"
        f"{schema_instruction(candidate_source=candidate_source, top_k=top_k)}"
    )


AGENT_SYSTEM_PROMPTS = {
    "BioAgent": (
        "You are BioAgent, a specialist in biological accessibility and topology constraints. "
        "Focus on transmembrane overlap, PTM occlusion risk, motif-functional caveats, and disulfide constraints. "
        "Return compact evidence-grounded votes and candidate adjustments using provided fields only. "
        "Return exactly one JSON object with double-quoted keys/strings and no markdown/preamble/postamble."
    ),
    "ChemAgent": (
        "You are ChemAgent, a specialist in local chemistry plausibility. "
        "Focus on hydropathy balance, side-chain composition, PTM chemical accessibility impacts, and motif-sensitive binding plausibility. "
        "Return compact evidence-grounded votes and candidate adjustments using provided fields only. "
        "Return exactly one JSON object with double-quoted keys/strings and no markdown/preamble/postamble."
    ),
    "RiskAgent": (
        "You are RiskAgent, a specialist in uncertainty and failure analysis. "
        "Focus on stacked PTM/motif risks, topology ambiguity, and downstream failure modes. "
        "Return compact evidence-grounded votes and candidate adjustments using provided fields only. "
        "Return exactly one JSON object with double-quoted keys/strings and no markdown/preamble/postamble."
    ),
    "DecisionAgent": (
        "You are DecisionAgent. Synthesize specialist critiques into a final modality decision and ranking. "
        "Return a compact final ranking grounded in specialist/context fields only. "
        "Return exactly one JSON object with double-quoted keys/strings and no markdown/preamble/postamble."
    ),
}


def build_specialist_prompt(agent_name: str, context_json: str) -> str:
    schema_hint = (
        '{"agent":"%s","modality_votes":{"epitope":0.0,"pocket":0.0,"other":0.0},'
        '"candidate_adjustments":[{"candidate_id":"...","delta":0.0,"reason":"...","evidence":["..."]}],'
        '"risk_flags":["..."],"summary":"..."}'
    ) % agent_name
    return (
        f"Role: {agent_name}\n"
        "Return only strict JSON matching this schema template exactly (keys may contain additional details):\n"
        f"{schema_hint}\n"
        "Required formatting rules: no markdown/code fences, no comments, no trailing commas, no extra text before/after JSON.\n"
        "Use only valid JSON literals (double quotes for all keys/strings; true/false/null for booleans/nulls).\n"
        "Never omit required keys.\n"
        "Do not output all-zero votes unless all evidence is missing. modality_votes must be normalized and sum to 1.0 (+/- 0.02).\n"
        "When candidates are provided, candidate_adjustments must include at least one row and no more than 8 rows.\n"
        "summary must be one short sentence. Each candidate reason must be one short sentence.\n"
        "If PTM sites are present, include PTM-aware wording. If motif hits are present, include motif-aware wording.\n"
        "Output must start with '{' and end with '}'. Do not include markdown fences or extra prose.\n"
        f"Input evidence JSON:\n{context_json}"
    )


def build_decision_prompt(context_json: str, top_k: int, requested_mode: str = "auto") -> str:
    forced_mode = str(requested_mode or "").strip().lower()
    forced_block = ""
    if forced_mode in {"epitope", "pocket"}:
        forced_block = (
            f"Requested modality is fixed to '{forced_mode}'. "
            "Set recommended_modality to this value and keep ranking limited to candidates in that modality.\n"
        )
    return (
        "Role: DecisionAgent\n"
        "Return only strict JSON with keys: "
        "recommended_modality, modality_confidence, ranking, global_risks.\n"
        f"{forced_block}"
        "Required formatting rules: no markdown/code fences, no comments, no trailing commas, no extra text before/after JSON.\n"
        "Use only valid JSON literals (double quotes for all keys/strings; true/false/null for booleans/nulls).\n"
        "Never omit required keys; use empty arrays if needed.\n"
        "ranking must be a list of {rank, candidate_id, reason, confidence_score, confidence_reason}.\n"
        f"ranking must contain at most {int(top_k)} rows and rank must start at 1 with no gaps.\n"
        "confidence_score must be in [0, 1] and represent your self-estimated confidence.\n"
        "ranking reasons and confidence_reason must be short evidence-grounded sentences and reference PTM/motif evidence when available.\n"
        "Output must start with '{' and end with '}'. Do not include markdown fences or extra prose.\n"
        f"Input evidence JSON:\n{context_json}"
    )
