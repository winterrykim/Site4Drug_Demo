#!/usr/bin/env python3
"""Shared constraint-first sequence feature utilities for Site4Drug."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from site4drug_inference.common.motif_features import scan_motifs, summarize_motif_hits
from site4drug_inference.common.ptm_backends import extract_ptm_sites

# Kyte-Doolittle hydropathy index
HYDROPATHY_INDEX = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

HYDROPHOBIC = set("AILMFWVP")
POLAR = set("STNQY")
CHARGED_POS = set("RKH")
CHARGED_NEG = set("DE")

NXS_PATTERN = re.compile(r"N[^P][ST]")

PTM_TYPE_N_LINKED = "N-linked_glycosylation"


@dataclass
class PTMSite:
    """PTM site with a masked neighborhood."""

    ptm_type: str
    position: int
    mask_start: int
    mask_end: int
    rule_confidence: str = "medium"
    source: str = "motif_rule"


@dataclass
class CandidateFeature:
    """Candidate region with derived sequence features and heuristic score."""

    candidate_id: str
    mode: str
    start: int
    end: int
    peptide: str
    mean_hydropathy: float
    hydrophobic_fraction: float
    polar_fraction: float
    positive_fraction: float
    negative_fraction: float
    cysteine_count: int
    overlaps_tm: bool
    overlaps_ptm_mask: bool
    ptm_overlap_by_type: dict[str, int]
    ptm_density: float
    motif_hits_overlapping: list[dict]
    motif_hit_count: int
    risk_flags: list[str]
    heuristic_score: float

    def as_dict(self) -> dict:
        return asdict(self)


def sliding_hydropathy(seq: str, window: int = 19) -> list[float]:
    """Compute sliding-window mean hydropathy (center-aligned)."""
    if not seq:
        return []
    vals = [HYDROPATHY_INDEX.get(a, 0.0) for a in seq]
    if len(vals) < window:
        mean_val = sum(vals) / max(len(vals), 1)
        return [mean_val] * len(vals)

    means = []
    running_sum = sum(vals[:window])
    means.append(running_sum / window)
    for i in range(1, len(vals) - window + 1):
        running_sum += vals[i + window - 1] - vals[i - 1]
        means.append(running_sum / window)

    pad_left = window // 2
    pad_right = len(vals) - len(means) - pad_left
    return [means[0]] * pad_left + means + [means[-1]] * pad_right


def find_tm_regions(
    hydropathy: list[float],
    threshold: float = 1.6,
    min_len: int = 15,
) -> list[tuple[int, int]]:
    """Heuristic transmembrane region detection from hydropathy profile."""
    regions: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(hydropathy):
        if value >= threshold:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_len:
                regions.append((start + 1, i))
            start = None
    if start is not None and (len(hydropathy) - start) >= min_len:
        regions.append((start + 1, len(hydropathy)))
    return regions


def _build_ptm_sites(
    seq: str,
    positions: Iterable[int],
    ptm_type: str,
    pad: int,
    rule_confidence: str,
    source: str = "motif_rule",
) -> list[PTMSite]:
    seq_len = len(seq)
    out: list[PTMSite] = []
    for pos in sorted({int(p) for p in positions if 1 <= int(p) <= seq_len}):
        out.append(
            PTMSite(
                ptm_type=ptm_type,
                position=pos,
                mask_start=max(1, pos - pad),
                mask_end=min(seq_len, pos + pad),
                rule_confidence=rule_confidence,
                source=source,
            )
        )
    return out


def find_glycosylation_sites(seq: str, pad: int = 5) -> list[PTMSite]:
    """Find N-linked glycosylation sequons (NxS/T) and apply mask padding."""
    positions = [match.start() + 1 for match in NXS_PATTERN.finditer(seq)]
    return _build_ptm_sites(
        seq,
        positions,
        ptm_type=PTM_TYPE_N_LINKED,
        pad=pad,
        rule_confidence="high",
    )


def find_phosphoser_thr_sites(seq: str, pad: int = 3) -> list[PTMSite]:
    """Rule-based phospho S/T motif detector."""
    positions: list[int] = []
    for i, aa in enumerate(seq):
        if aa not in {"S", "T"}:
            continue
        prev1 = seq[i - 1] if i - 1 >= 0 else ""
        prev2 = seq[i - 2] if i - 2 >= 0 else ""
        next1 = seq[i + 1] if i + 1 < len(seq) else ""
        if next1 == "P" or next1 in {"D", "E"} or prev1 in {"R", "K"} or prev2 in {"R", "K"}:
            positions.append(i + 1)
    return _build_ptm_sites(
        seq,
        positions,
        ptm_type="Phosphoserine_Phosphothreonine",
        pad=pad,
        rule_confidence="medium",
    )


def find_ubiquitination_sites(seq: str, pad: int = 4) -> list[PTMSite]:
    """Rule-based lysine-centric ubiquitination motif detector."""
    positions: list[int] = []
    for i, aa in enumerate(seq):
        if aa != "K":
            continue
        win = seq[max(0, i - 2) : min(len(seq), i + 3)]
        neighbors = win.replace("K", "")
        if any(ch in "DESTQ" for ch in neighbors):
            positions.append(i + 1)
    return _build_ptm_sites(
        seq,
        positions,
        ptm_type="Ubiquitination",
        pad=pad,
        rule_confidence="low",
    )


def find_n6_acetyllysine_sites(seq: str, pad: int = 4) -> list[PTMSite]:
    """Rule-based N6-acetyllysine motif detector."""
    positions: list[int] = []
    for i, aa in enumerate(seq):
        if aa != "K":
            continue
        prev1 = seq[i - 1] if i - 1 >= 0 else ""
        next1 = seq[i + 1] if i + 1 < len(seq) else ""
        if i < 15 or prev1 in "ASGT" or next1 in "ASGT":
            positions.append(i + 1)
    return _build_ptm_sites(
        seq,
        positions,
        ptm_type="N6-acetyllysine",
        pad=pad,
        rule_confidence="low",
    )


def find_methylarginine_sites(seq: str, pad: int = 3) -> list[PTMSite]:
    """Rule-based methylarginine motif detector (RG/RGG-like contexts)."""
    positions: list[int] = []
    for i, aa in enumerate(seq):
        if aa != "R":
            continue
        prev1 = seq[i - 1] if i - 1 >= 0 else ""
        next1 = seq[i + 1] if i + 1 < len(seq) else ""
        next2 = seq[i + 2] if i + 2 < len(seq) else ""
        if next1 == "G" or (next1 == "G" and next2 == "G") or prev1 == "G":
            positions.append(i + 1)
    return _build_ptm_sites(
        seq,
        positions,
        ptm_type="Methylarginine",
        pad=pad,
        rule_confidence="low",
    )


def find_hydroxyproline_sites(seq: str, pad: int = 3) -> list[PTMSite]:
    """Rule-based hydroxyproline motif detector."""
    positions: list[int] = []
    for i, aa in enumerate(seq):
        if aa != "P":
            continue
        prev1 = seq[i - 1] if i - 1 >= 0 else ""
        next1 = seq[i + 1] if i + 1 < len(seq) else ""
        if prev1 in "GAS" or next1 in "GAS":
            positions.append(i + 1)
    return _build_ptm_sites(
        seq,
        positions,
        ptm_type="Hydroxyproline",
        pad=pad,
        rule_confidence="low",
    )


def find_pyrrolidone_carboxylic_acid_sites(seq: str, pad: int = 1) -> list[PTMSite]:
    """Rule-based pyroglutamate-like N-terminal conversion detector."""
    positions: list[int] = []
    if seq and seq[0] in {"Q", "E"}:
        positions.append(1)
    return _build_ptm_sites(
        seq,
        positions,
        ptm_type="Pyrrolidone_carboxylic_acid",
        pad=pad,
        rule_confidence="medium",
    )


def _find_ptm_sites_rulepack(seq: str, pad: int = 5, source: str = "multi_rule") -> list[PTMSite]:
    """Find PTM sites based on selected source policy."""
    source_norm = str(source or "multi_rule").strip().lower()
    if source_norm == "glyco_only":
        return find_glycosylation_sites(seq, pad=pad)

    all_sites: list[PTMSite] = []
    all_sites.extend(find_glycosylation_sites(seq, pad=pad))
    all_sites.extend(find_phosphoser_thr_sites(seq, pad=max(2, pad - 2)))
    all_sites.extend(find_ubiquitination_sites(seq, pad=max(3, pad - 1)))
    all_sites.extend(find_n6_acetyllysine_sites(seq, pad=max(3, pad - 1)))
    all_sites.extend(find_methylarginine_sites(seq, pad=max(2, pad - 2)))
    all_sites.extend(find_hydroxyproline_sites(seq, pad=max(2, pad - 2)))
    all_sites.extend(find_pyrrolidone_carboxylic_acid_sites(seq, pad=1))

    dedup: dict[tuple[str, int], PTMSite] = {}
    for site in all_sites:
        key = (site.ptm_type, site.position)
        if key not in dedup:
            dedup[key] = site
    sites = list(dedup.values())
    sites.sort(key=lambda s: (s.position, s.ptm_type))
    return sites


def find_ptm_sites(seq: str, pad: int = 5, source: str = "multi_rule") -> list[PTMSite]:
    """Backwards-compatible PTM finder for local rule-pack policies."""
    return _find_ptm_sites_rulepack(seq, pad=pad, source=source)


def summarize_ptm_sites(ptm_sites: list[PTMSite]) -> dict:
    counts: dict[str, int] = {}
    masked_positions: set[int] = set()
    for site in ptm_sites:
        counts[site.ptm_type] = counts.get(site.ptm_type, 0) + 1
        masked_positions.update(range(site.mask_start, site.mask_end + 1))
    return {
        "total_sites": len(ptm_sites),
        "counts_by_type": counts,
        "n_masked_positions": len(masked_positions),
    }


def find_cysteines(seq: str) -> list[int]:
    """Find cysteine positions (1-indexed)."""
    return [i + 1 for i, aa in enumerate(seq) if aa == "C"]


def region_hydropathy(seq: str, start: int, end: int) -> float:
    """Mean region hydropathy for a 1-indexed inclusive span."""
    region = seq[start - 1 : end]
    if not region:
        return 0.0
    return sum(HYDROPATHY_INDEX.get(a, 0.0) for a in region) / len(region)


def region_overlaps(
    start: int,
    end: int,
    intervals: Iterable[tuple[int, int]],
) -> bool:
    """Whether [start, end] overlaps any 1-indexed interval."""
    for s, e in intervals:
        if not (end < s or start > e):
            return True
    return False


def count_in_region(start: int, end: int, positions: Iterable[int]) -> int:
    """Count positions contained by a 1-indexed inclusive span."""
    return sum(1 for pos in positions if start <= pos <= end)


def aa_composition(peptide: str) -> dict[str, float]:
    """Amino-acid class composition fractions."""
    n = len(peptide) or 1
    return {
        "hydrophobic_fraction": sum(1 for a in peptide if a in HYDROPHOBIC) / n,
        "polar_fraction": sum(1 for a in peptide if a in POLAR) / n,
        "positive_fraction": sum(1 for a in peptide if a in CHARGED_POS) / n,
        "negative_fraction": sum(1 for a in peptide if a in CHARGED_NEG) / n,
    }


def aa_composition_str(peptide: str) -> str:
    comp = aa_composition(peptide)
    return (
        f"hydrophobic {comp['hydrophobic_fraction']:.0%}, "
        f"polar {comp['polar_fraction']:.0%}, "
        f"charged(+) {comp['positive_fraction']:.0%}, "
        f"charged(-) {comp['negative_fraction']:.0%}"
    )


def _ptm_overlap_by_type(start: int, end: int, ptm_sites: list[PTMSite]) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    for site in ptm_sites:
        if end < site.mask_start or start > site.mask_end:
            continue
        overlaps[site.ptm_type] = overlaps.get(site.ptm_type, 0) + 1
    return overlaps


def _motif_hits_in_region(start: int, end: int, motif_hits: list[dict]) -> list[dict]:
    hits: list[dict] = []
    for hit in motif_hits:
        m_start = int(hit.get("start", 0))
        m_end = int(hit.get("end", m_start))
        if m_end < start or m_start > end:
            continue
        hits.append(
            {
                "motif_name": hit.get("motif_name", "Motif"),
                "pattern_id": hit.get("pattern_id", ""),
                "start": m_start,
                "end": m_end,
                "evidence_source": hit.get("evidence_source", "remote_scanprosite_biopython"),
            }
        )
    return hits


def _risk_flags_for_region(
    mean_h: float,
    overlaps_tm: bool,
    ptm_overlap_by_type: dict[str, int],
    ptm_density: float,
    cysteine_count: int,
    motif_hits_overlapping: list[dict],
) -> list[str]:
    flags = []
    total_ptm = sum(ptm_overlap_by_type.values())
    if overlaps_tm:
        flags.append("TM-overlap")
    if total_ptm > 0:
        flags.append("PTM-overlap")
    if ptm_overlap_by_type.get(PTM_TYPE_N_LINKED, 0) > 0:
        flags.append("glyco-mask-overlap")
    if ptm_density >= 0.25 or total_ptm >= 3:
        flags.append("PTM-dense")
        flags.append("PTM-clustered")
    if cysteine_count >= 2:
        flags.append("disulfide-constrained")
    if mean_h > 1.0:
        flags.append("hydrophobic-core")
    if motif_hits_overlapping:
        flags.append("motif-overlap")
        motif_names = " ".join(str(h.get("motif_name", "")) for h in motif_hits_overlapping).lower()
        if "zinc" in motif_names:
            flags.append("zinc-finger-like-motif")
        if "dna" in motif_names or "nls" in motif_names:
            flags.append("basic-functional-motif")
    return flags


def _ptm_penalty(
    mode: str,
    ptm_overlap_by_type: dict[str, int],
    ptm_density: float,
    ptm_policy: str,
) -> float:
    total = sum(ptm_overlap_by_type.values())
    if total <= 0:
        return 0.0

    policy = str(ptm_policy or "tiered").strip().lower()
    if policy == "hard":
        return 2.5 if mode != "pocket" else 1.5
    if policy == "soft":
        return 0.18 * total + 0.25 * ptm_density

    # tiered (default)
    glyco = ptm_overlap_by_type.get(PTM_TYPE_N_LINKED, 0)
    other = max(total - glyco, 0)
    glyco_weight = 0.95 if mode != "pocket" else 0.45
    penalty = glyco_weight * glyco + 0.12 * other
    if ptm_density >= 0.25:
        penalty += 0.25
    return penalty


def _heuristic_score(
    mode: str,
    mean_h: float,
    overlaps_tm: bool,
    comp: dict[str, float],
    ptm_penalty: float,
    motif_hits_overlapping: list[dict],
) -> float:
    """Simple deterministic score used for fallback ranking."""
    motif_count = len(motif_hits_overlapping)
    motif_names = " ".join(str(m.get("motif_name", "")) for m in motif_hits_overlapping).lower()

    if mode == "pocket":
        score = mean_h + 1.0 * comp["hydrophobic_fraction"] - 0.4 * comp["negative_fraction"]
        score -= 0.45 * ptm_penalty
        if motif_count > 0 and ("p_loop" in motif_names or "zipper" in motif_names):
            score += 0.08
        return score

    # epitope/default mode
    score = -abs(mean_h + 0.8)
    score += 0.4 * comp["polar_fraction"] + 0.2 * comp["positive_fraction"]
    score -= 1.0 if overlaps_tm else 0.0
    score -= ptm_penalty
    score -= 0.04 * motif_count
    if "zinc" in motif_names or "dna" in motif_names:
        score -= 0.12
    return score


def _topology_accessibility_labels(
    *,
    mean_h: float,
    overlaps_tm: bool,
    tm_threshold: float,
) -> tuple[str, str, float]:
    """Infer coarse topology/accessibility label and confidence from hydropathy + TM overlap."""
    if overlaps_tm:
        label = "tmd"
        accessibility = "restricted"
        distance = max(mean_h - tm_threshold, 0.0)
        confidence = min(0.95, 0.62 + 0.18 * distance)
        return label, accessibility, round(confidence, 3)

    label = "outside"
    accessibility = "exposed"
    distance = max(tm_threshold - mean_h, 0.0)
    confidence = min(0.95, 0.58 + 0.12 * distance)
    return label, accessibility, round(confidence, 3)


def _modality_flags(mode: str) -> list[str]:
    mode_norm = str(mode or "").strip().lower()
    if mode_norm == "pocket":
        return ["pocket-preferred", "sequence-patch-proxy"]
    if mode_norm == "epitope":
        return ["epitope-preferred"]
    return ["modality-unknown"]


def _patch_descriptor(mode: str, start: int, end: int) -> dict | None:
    if str(mode or "").strip().lower() != "pocket":
        return None
    center = (start + end) // 2
    return {
        "type": "sequence_patch_proxy",
        "start": start,
        "end": end,
        "center_residue": center,
        "length": end - start + 1,
        "note": "Structure not supplied; sequence-defined proxy patch.",
    }


def _iter_candidate_spans(seq_len: int, mode: str, stride: int) -> Iterable[tuple[int, int]]:
    if mode == "pocket":
        lengths = (10, 12, 14, 16)
    else:
        lengths = (12, 15, 18, 20)
    for length in lengths:
        if length > seq_len:
            continue
        for start in range(1, seq_len - length + 2, stride):
            end = start + length - 1
            yield start, end


def build_candidate_features(
    seq: str,
    mode: str,
    top_n: int = 120,
    stride: int = 5,
    hydropathy_window: int = 19,
    tm_threshold: float = 1.6,
    tm_min_len: int = 15,
    ptm_pad: int = 5,
    ptm_source: str = "musitedeep",
    ptm_policy: str = "tiered",
    motif_source: str = "remote",
    use_motif: bool = True,
    musitedeep_api_base_url: str = "https://www.musite.net",
    musitedeep_model_map: str | None = None,
    strict_ptm_backend: bool = False,
) -> dict[str, object]:
    """Generate and rank heuristic candidate regions from sequence-level constraints."""
    seq = seq.strip().upper()
    hydropathy_profile = sliding_hydropathy(seq, window=hydropathy_window)
    tm_regions = find_tm_regions(hydropathy_profile, threshold=tm_threshold, min_len=tm_min_len)

    ptm_backend = extract_ptm_sites(
        seq=seq,
        pad=ptm_pad,
        source=ptm_source,
        site_builder=PTMSite,
        multi_rule_finder=_find_ptm_sites_rulepack,
        glyco_only_finder=_find_ptm_sites_rulepack,
        musitedeep_api_base_url=musitedeep_api_base_url,
        musitedeep_model_map=musitedeep_model_map,
        strict=bool(strict_ptm_backend),
    )
    ptm_sites = ptm_backend.get("ptm_sites", [])
    ptm_masks = [(site.mask_start, site.mask_end) for site in ptm_sites]
    ptm_summary = dict(ptm_backend.get("ptm_summary", summarize_ptm_sites(ptm_sites)))

    motif_hits, motif_meta = scan_motifs(
        seq,
        source=motif_source,
        enabled=use_motif,
    )
    motif_summary = summarize_motif_hits(motif_hits)

    cysteine_positions = find_cysteines(seq)

    candidates: list[CandidateFeature] = []
    for idx, (start, end) in enumerate(_iter_candidate_spans(len(seq), mode=mode, stride=stride), start=1):
        peptide = seq[start - 1 : end]
        mean_h = region_hydropathy(seq, start, end)
        overlaps_tm = region_overlaps(start, end, tm_regions)
        ptm_overlap_by_type = _ptm_overlap_by_type(start, end, ptm_sites)
        overlaps_ptm = region_overlaps(start, end, ptm_masks)
        ptm_total = sum(ptm_overlap_by_type.values())
        ptm_density = ptm_total / max(end - start + 1, 1)

        motif_hits_overlapping = _motif_hits_in_region(start, end, motif_hits)
        motif_hit_count = len(motif_hits_overlapping)

        cys = count_in_region(start, end, cysteine_positions)
        comp = aa_composition(peptide)

        flags = _risk_flags_for_region(
            mean_h=mean_h,
            overlaps_tm=overlaps_tm,
            ptm_overlap_by_type=ptm_overlap_by_type,
            ptm_density=ptm_density,
            cysteine_count=cys,
            motif_hits_overlapping=motif_hits_overlapping,
        )
        ptm_penalty = _ptm_penalty(
            mode=mode,
            ptm_overlap_by_type=ptm_overlap_by_type,
            ptm_density=ptm_density,
            ptm_policy=ptm_policy,
        )
        score = _heuristic_score(
            mode=mode,
            mean_h=mean_h,
            overlaps_tm=overlaps_tm,
            comp=comp,
            ptm_penalty=ptm_penalty,
            motif_hits_overlapping=motif_hits_overlapping,
        )

        candidates.append(
            CandidateFeature(
                candidate_id=f"C{idx:04d}",
                mode=mode,
                start=start,
                end=end,
                peptide=peptide,
                mean_hydropathy=mean_h,
                hydrophobic_fraction=comp["hydrophobic_fraction"],
                polar_fraction=comp["polar_fraction"],
                positive_fraction=comp["positive_fraction"],
                negative_fraction=comp["negative_fraction"],
                cysteine_count=cys,
                overlaps_tm=overlaps_tm,
                overlaps_ptm_mask=overlaps_ptm,
                ptm_overlap_by_type=ptm_overlap_by_type,
                ptm_density=round(ptm_density, 4),
                motif_hits_overlapping=motif_hits_overlapping,
                motif_hit_count=motif_hit_count,
                risk_flags=flags,
                heuristic_score=score,
            )
        )

    candidates.sort(key=lambda c: c.heuristic_score, reverse=True)
    selected = candidates[: max(top_n, 1)]
    return {
        "sequence_length": len(seq),
        "hydropathy_profile": hydropathy_profile,
        "tm_regions": tm_regions,
        "ptm_sites": [asdict(site) for site in ptm_sites],
        "ptm_summary": ptm_summary,
        "motif_hits": motif_hits,
        "motif_summary": motif_summary,
        "cysteine_positions": cysteine_positions,
        "feature_provenance": {
            "module": "site4drug_inference.common.constraint_features",
            "ptm_source": str(ptm_source or "multi_rule"),
            "ptm_rule_version": str(
                (ptm_backend.get("backend", {}) or {}).get("ptm_rule_version", "rulepack_v1")
            ),
            "ptm_backend_selected": str((ptm_backend.get("backend", {}) or {}).get("ptm_backend_selected", ptm_source)),
            "ptm_backend_effective": str((ptm_backend.get("backend", {}) or {}).get("ptm_backend_effective", ptm_source)),
            "musitedeep_available": bool((ptm_backend.get("backend", {}) or {}).get("musitedeep_available", False)),
            "musitedeep_status": str((ptm_backend.get("backend", {}) or {}).get("musitedeep_status", "not_requested")),
            "musitedeep_models_attempted": list(
                (ptm_backend.get("backend", {}) or {}).get("musitedeep_models_attempted", [])
            ),
            "musitedeep_models_succeeded": list(
                (ptm_backend.get("backend", {}) or {}).get("musitedeep_models_succeeded", [])
            ),
            "musitedeep_api_base_url": str(
                (ptm_backend.get("backend", {}) or {}).get("musitedeep_api_base_url", "")
            ),
            "musitedeep_endpoint_urls": list(
                (ptm_backend.get("backend", {}) or {}).get("musitedeep_endpoint_urls", [])
            ),
            "musitedeep_error_summary": str(
                (ptm_backend.get("backend", {}) or {}).get("musitedeep_error_summary", "")
            ),
            "ptm_policy": str(ptm_policy or "tiered"),
            "motif_source": motif_meta.get("motif_source", str(motif_source or "local")),
            "motif_library_version": motif_meta.get("motif_library_version", "unknown"),
            "motif_remote_status": motif_meta.get("motif_remote_status", "not_requested"),
            "hydropathy_window": hydropathy_window,
            "tm_threshold": tm_threshold,
            "tm_min_len": tm_min_len,
            "ptm_mask_pad": ptm_pad,
        },
        "candidates": [c.as_dict() for c in selected],
        "ptm_backend": {
            **(ptm_backend.get("backend", {}) or {}),
            "warnings": list(ptm_backend.get("warnings", [])),
        },
        "api_raw": {
            "musitedeep": {
                "status": str((ptm_backend.get("backend", {}) or {}).get("musitedeep_status", "not_requested")),
                "api_base_url": str((ptm_backend.get("backend", {}) or {}).get("musitedeep_api_base_url", "")),
                "endpoint_urls": list((ptm_backend.get("backend", {}) or {}).get("musitedeep_endpoint_urls", [])),
                "errors": str((ptm_backend.get("backend", {}) or {}).get("musitedeep_error_summary", "")),
                "raw_calls": list((ptm_backend.get("backend", {}) or {}).get("musitedeep_raw_calls", [])),
            },
            "scanprosite": {
                "status": str(motif_meta.get("motif_remote_status", "not_requested")),
                "warning": str(motif_meta.get("motif_remote_warning", "") or ""),
                "error": str(motif_meta.get("motif_remote_error", "") or ""),
                "raw_xml": str(motif_meta.get("motif_remote_raw_xml", "") or ""),
                "n_hits": int(len(motif_hits)),
            },
        },
    }


def slice_sequence_with_overlap(sequence: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Split sequence into overlapped chunks; returns (start,end,chunk_seq) 1-indexed."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sequence = sequence.strip().upper()
    chunks: list[tuple[int, int, str]] = []
    step = chunk_size - overlap
    start = 0
    while start < len(sequence):
        end = min(start + chunk_size, len(sequence))
        chunk = sequence[start:end]
        chunks.append((start + 1, end, chunk))
        if end == len(sequence):
            break
        start += step
    return chunks
