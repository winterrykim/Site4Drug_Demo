#!/usr/bin/env python3
"""Plot helpers for Site4Drug run artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "site4drug_mpl_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
if "XDG_CACHE_HOME" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "site4drug_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

MODE_COLORS = {
    "epitope": "#2e8b57",
    "pocket": "#3b6fb6",
    "other": "#7a8288",
}


def _ptm_type_order(ptm_sites: list[dict]) -> list[str]:
    seen: list[str] = []
    for site in ptm_sites:
        ptm_type = site.get("ptm_type", "PTM")
        if ptm_type not in seen:
            seen.append(ptm_type)
    return seen or ["PTM"]


def render_hydropathy_ptm_candidate_plot(
    sequence: str,
    hydropathy_profile: list[float],
    ptm_sites: list[dict],
    ranked_candidates: list[dict],
    output_png: Path,
    output_json: Path,
    tm_threshold: float = 1.6,
) -> dict:
    """Render a 3-panel figure with hydropathy, PTM dots, and candidate tracks."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(1, len(sequence) + 1))
    ptm_types = _ptm_type_order(ptm_sites)
    ptm_y_index = {name: idx for idx, name in enumerate(ptm_types)}

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.6, 2.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Panel 1: hydropathy profile
    ax1.plot(x[: len(hydropathy_profile)], hydropathy_profile, lw=2.0, label="Mean HI (window=19)")
    ax1.axhline(tm_threshold, ls="--", lw=1.8, label=f"Threshold = {tm_threshold:.1f}")
    ax1.set_ylabel("Mean hydropathy index")
    ax1.set_title("Hydropathy plot + PTM dot plot + candidate tracks")
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.legend(loc="upper right")

    # Panel 2: PTM dots
    color_cycle = [
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#ff7f0e",
        "#1f77b4",
        "#d62728",
    ]
    color_map = {
        ptm: color_cycle[idx % len(color_cycle)] for idx, ptm in enumerate(ptm_types)
    }
    confidence_marker = {
        "high": "o",
        "medium": "^",
        "low": "x",
    }
    for site in ptm_sites:
        ptm = site.get("ptm_type", "PTM")
        pos = int(site.get("position", 0))
        conf = str(site.get("rule_confidence", "medium")).lower()
        marker = confidence_marker.get(conf, "^")
        ax2.scatter(pos, ptm_y_index[ptm], s=42, c=color_map[ptm], alpha=0.95, marker=marker)
    ax2.set_yticks(list(ptm_y_index.values()))
    ax2.set_yticklabels(ptm_types)
    ax2.set_ylabel("PTM pattern")
    ax2.grid(alpha=0.2, linestyle=":")
    type_handles = [
        mlines.Line2D([], [], color=color_map[ptm], marker="o", linestyle="None", markersize=7, label=ptm)
        for ptm in ptm_types
    ]
    conf_handles = [
        mlines.Line2D([], [], color="#374151", marker=confidence_marker["high"], linestyle="None", markersize=7, label="PTM confidence: high"),
        mlines.Line2D([], [], color="#374151", marker=confidence_marker["medium"], linestyle="None", markersize=7, label="PTM confidence: medium"),
        mlines.Line2D([], [], color="#374151", marker=confidence_marker["low"], linestyle="None", markersize=7, label="PTM confidence: low"),
    ]
    ax2.legend(handles=[*type_handles, *conf_handles], loc="upper right", fontsize=8)

    # Panel 3: candidate bars
    y_labels: list[str] = []
    seen_modes: set[str] = set()
    for idx, cand in enumerate(ranked_candidates):
        rank = cand.get("rank", idx + 1)
        mode = str(cand.get("mode", "other")).lower()
        if mode not in MODE_COLORS:
            mode = "other"
        seen_modes.add(mode)
        label = f"Candidate {rank} ({mode})"
        y_labels.append(label)
        start = int(cand.get("start", 1))
        end = int(cand.get("end", start))
        width = max(1, end - start + 1)
        flags = cand.get("flags", []) or cand.get("risk_flags", [])
        flag_text = " ".join(str(flag) for flag in flags)
        has_ptm_flag = "PTM" in flag_text
        is_ptm_dense = ("PTM-dense" in flag_text) or ("PTM-clustered" in flag_text)
        color = MODE_COLORS.get(mode, MODE_COLORS["other"])
        ax3.barh(idx, width, left=start, color=color, alpha=0.95, edgecolor="white")
        if has_ptm_flag:
            label = "PTM dense" if is_ptm_dense else "PTM overlap"
            ax3.text(
                start + max(width // 4, 1),
                idx + 0.08,
                label,
                fontsize=9,
                color="gold",
                fontweight="bold",
            )

    ax3.set_yticks(range(len(y_labels)))
    ax3.set_yticklabels(y_labels)
    ax3.set_xlabel(f"Position (1-{len(sequence)})")
    ax3.set_ylabel("Ranked candidates")
    ax3.grid(alpha=0.25, linestyle=":")
    ax3.invert_yaxis()

    # Legend clarifies bar color semantics + PTM annotation.
    mode_order = ("epitope", "pocket", "other")
    mode_handles = [
        mpatches.Patch(color=MODE_COLORS[m], label=f"{m} candidate")
        for m in mode_order
        if m in seen_modes
    ]
    if not mode_handles:
        mode_handles = [mpatches.Patch(color=MODE_COLORS["other"], label="candidate")]
    ptm_handle = mlines.Line2D(
        [],
        [],
        color="gold",
        marker="*",
        linestyle="None",
        markersize=10,
        label="PTM risk annotation",
    )
    legend = ax3.legend(
        handles=[*mode_handles, ptm_handle],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )
    legend.get_frame().set_alpha(0.95)

    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "sequence_length": len(sequence),
        "tm_threshold": tm_threshold,
        "n_ptm_sites": len(ptm_sites),
        "n_ranked_candidates": len(ranked_candidates),
        "candidate_mode_colors": MODE_COLORS,
        "candidate_mode_legend": {
            "epitope": "emerald green bars",
            "pocket": "steel blue bars",
            "other": "slate gray bars",
        },
        "ptm_risk_annotation": "gold text on candidate track (PTM overlap or PTM dense)",
        "ptm_confidence_markers": {
            "high": "o",
            "medium": "^",
            "low": "x",
        },
        "ptm_sites": ptm_sites,
        "ranked_candidates": ranked_candidates,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "plot_png": str(output_png),
        "plot_json": str(output_json),
    }
