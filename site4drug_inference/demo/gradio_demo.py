#!/usr/bin/env python3
"""Gradio UI demo for Site4Drug inference."""

from __future__ import annotations

import os
import queue
from pathlib import Path
import sys
import threading
import time
from typing import Any

import gradio as gr
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from site4drug_inference.common.env_utils import ensure_tinker_api_key
from site4drug_inference.demo import predict_site
from site4drug_inference.demo.ui_helpers import (
    agent_conclusion_md,
    build_analysis_json,
    build_artifacts_text,
    load_report_markdown,
    ranked_candidates_display_df,
    resolve_demo_input_sequence,
    resolve_plot_path,
)


def run_demo(
    uniprot: str,
    input_mode: str,
    sequence_text: str,
    sequence_file: str | None,
    mode: str,
    top_k: int,
    self_consistency_k: int,
    sampling_seed: str,
    ptm_source: str,
    ptm_policy: str,
    use_motif: bool,
    use_base_model: bool,
    base_model: str,
    checkpoint: str,
    max_tokens: int,
    max_input_tokens: int,
    output_dir: str,
    use_iedb_validation: bool,
    no_online_lookup: bool,
) -> Any:
    empty_df = pd.DataFrame(
        columns=["rank", "candidate_id", "mode", "peptide", "position", "confidence", "score", "source", "flags", "reason"]
    )
    progress_lines: list[str] = []
    current_step = "Initializing run"
    event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    state: dict[str, Any] = {"done": False, "result": None, "error": None}

    def _push_progress(event: dict[str, Any]) -> None:
        event_queue.put(dict(event))

    def _parse_sampling_seed(raw_value: str) -> int | None:
        text = str(raw_value or "").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError("Sampling seed must be an integer or blank.") from exc

    def _worker() -> None:
        try:
            parsed_sampling_seed = _parse_sampling_seed(sampling_seed)
            raw_sequence, input_source = resolve_demo_input_sequence(
                uniprot=uniprot,
                input_mode=input_mode,
                sequence_text=sequence_text,
                sequence_file=sequence_file,
                allow_online_lookup=not bool(no_online_lookup),
                paste_source_label="gradio_sequence_text",
                upload_source_prefix="gradio_upload:",
            )
            state["result"] = predict_site.run_prediction(
                uniprot=uniprot,
                raw_sequence=raw_sequence,
                checkpoint=None if use_base_model else checkpoint,
                base_model=base_model,
                mode=mode,
                candidate_source="llm_propose",
                top_k=int(top_k),
                max_tokens=int(max_tokens),
                temperature=0.0,
                self_consistency_k=int(self_consistency_k),
                sampling_seed=parsed_sampling_seed,
                max_input_tokens=int(max_input_tokens),
                output_dir=Path(output_dir),
                require_api_key=True,
                input_source=input_source,
                enable_plot=True,
                use_multi_agent=True,
                repair_with_base_model=True,
                panel_with_base_model=True,
                ptm_source=ptm_source,
                ptm_policy=ptm_policy,
                motif_source="remote",
                use_motif=bool(use_motif),
                musitedeep_api_base_url=predict_site.DEFAULT_MUSITEDEEP_API_BASE_URL,
                musitedeep_model_map=None,
                use_iedb_validation=bool(use_iedb_validation),
                iedb_table_path=predict_site.DEFAULT_IEDB_TABLE_PATH,
                iedb_iou_threshold=predict_site.DEFAULT_IEDB_IOU_THRESHOLD,
                progress_callback=_push_progress,
            )
        except Exception as exc:
            state["error"] = str(exc)
        finally:
            state["done"] = True

    threading.Thread(target=_worker, daemon=True).start()
    while not state["done"] or not event_queue.empty():
        while True:
            try:
                evt = event_queue.get_nowait()
            except queue.Empty:
                break
            label = str(evt.get("label", evt.get("step_key", "step")))
            status = str(evt.get("status", "running"))
            current_step = f"{label} [{status}]"
            progress_lines.append(f"- `{evt.get('timestamp_utc', '')}` `{evt.get('event_type', '')}` `{label}` -> `{status}`")
            if len(progress_lines) > 60:
                progress_lines = progress_lines[-60:]
        status_md = (
            "### Run in progress\n\n"
            f"- **Current step:** `{current_step}`\n"
            f"- **Progress events:** `{len(progress_lines)}`"
        )
        yield (
            status_md,
            f"**Current step:** `{current_step}`",
            "\n".join(progress_lines) or "- Waiting for events...",
            empty_df,
            None,
            {},
            "- Run in progress.",
            "",
            "",
        )
        time.sleep(0.15)

    if state["error"]:
        err = str(state["error"])
        yield (
            f"### Run failed\n\n`{err}`",
            f"**Current step:** `Failed`",
            "\n".join(progress_lines) or "- No progress events captured.",
            empty_df,
            None,
            {"error": err},
            "- Run failed before agent conclusions could be produced.",
            "",
            "",
        )
        return

    result = state.get("result", {}) or {}
    payload = result.get("run_payload", {}) or {}
    generation = payload.get("generation", {}) or {}
    plot_path = resolve_plot_path(result, payload)
    status_md = (
        f"### Run complete\n\n"
        f"- **Run status:** `{payload.get('run_status', 'unknown')}`\n"
        f"- **Recommended modality:** `{payload.get('recommended_modality', 'unknown')}` "
        f"(confidence `{float(payload.get('modality_confidence', 0.0) or 0.0):.2f}`)\n"
        f"- **Token strategy:** `{payload.get('token_strategy_used', 'unknown')}`\n"
        f"- **Candidate source:** `{generation.get('candidate_source', 'unknown')}`\n"
        f"- **Sampling seed:** `{generation.get('sampling_seed', None)}`\n"
        f"- **Parsed candidates:** `{int(payload.get('parsed_candidate_count', 0) or 0)}`\n"
        f"- **Output directory:** `{result.get('run_dir')}`"
    )
    analysis_json = build_analysis_json(payload)
    artifacts_text = build_artifacts_text(result, payload)
    report_md = load_report_markdown(result)

    yield (
        status_md,
        "**Current step:** `Completed`",
        "\n".join(progress_lines) or "- No progress events captured.",
        ranked_candidates_display_df(payload),
        plot_path,
        analysis_json,
        agent_conclusion_md(payload),
        artifacts_text,
        report_md,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Site4Drug Gradio Demo") as app:
        gr.Markdown(
            """
            # Site4Drug Demo (Gradio)
            Interactive demo for sequence input, mode selection, ranked candidates, and audit-style analysis.
            """
        )

        with gr.Row():
            uniprot = gr.Textbox(label="UniProt / Label", value="P29996")
            input_mode = gr.Radio(
                ["Paste sequence", "Upload FASTA", "UniProt lookup only"],
                value="Paste sequence",
                label="Sequence input mode",
            )

        with gr.Row():
            sequence_text = gr.Textbox(label="Sequence (plain or FASTA)", lines=8, placeholder="Paste amino-acid sequence or FASTA.")
            sequence_file = gr.File(label="Upload FASTA/TXT", file_types=[".fasta", ".fa", ".txt"], type="filepath")

        with gr.Accordion("Inference controls", open=True):
            with gr.Row():
                mode = gr.Dropdown(["auto", "epitope", "pocket"], value="auto", label="Mode")
                top_k = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Top-K")
                self_consistency_k = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Self-consistency K",
                )
            with gr.Row():
                sampling_seed = gr.Textbox(
                    label="Sampling seed (optional)",
                    value="42",
                    placeholder="Use blank for default backend behavior",
                )
            with gr.Row():
                ptm_source = gr.Dropdown(["musitedeep", "hybrid", "multi_rule", "glyco_only"], value="musitedeep", label="PTM source")
                ptm_policy = gr.Dropdown(["tiered", "hard", "soft"], value="tiered", label="PTM policy")
                use_motif = gr.Checkbox(value=True, label="Enable motif scan")
            with gr.Row():
                use_base_model = gr.Checkbox(value=False, label="Use base model only (no checkpoint)")
                base_model = gr.Textbox(label="Base model", value=predict_site.BASE_MODEL)
                checkpoint = gr.Textbox(label="Checkpoint", value=predict_site.DEFAULT_CHECKPOINT)
            with gr.Row():
                max_tokens = gr.Slider(minimum=256, maximum=12000, value=3000, step=256, label="Max output tokens")
                max_input_tokens = gr.Slider(minimum=2000, maximum=30000, value=10000, step=500, label="Max input tokens")
            with gr.Row():
                output_dir = gr.Textbox(label="Output directory", value=str(predict_site.DEFAULT_OUTPUT_DIR))
                use_iedb_validation = gr.Checkbox(value=True, label="Enable IEDB validation")
                no_online_lookup = gr.Checkbox(value=False, label="Disable online UniProt lookup")

        run_btn = gr.Button("Run Site4Drug", variant="primary")

        status_md = gr.Markdown()
        current_step_md = gr.Markdown()
        progress_log_md = gr.Markdown()
        ranked_df = gr.Dataframe(label="Ranked candidates", interactive=False)
        plot_img = gr.Image(label="Hydropathy + PTM + Candidate Tracks", type="filepath")
        analysis_json = gr.JSON(label="Summaries and provenance")
        agent_md = gr.Markdown(label="Agent conclusions")
        artifacts_box = gr.Textbox(label="Artifact paths", lines=4)
        report_md = gr.Textbox(label="Report markdown preview", lines=14)

        run_btn.click(
            fn=run_demo,
            inputs=[
                uniprot,
                input_mode,
                sequence_text,
                sequence_file,
                mode,
                top_k,
                self_consistency_k,
                sampling_seed,
                ptm_source,
                ptm_policy,
                use_motif,
                use_base_model,
                base_model,
                checkpoint,
                max_tokens,
                max_input_tokens,
                output_dir,
                use_iedb_validation,
                no_online_lookup,
            ],
            outputs=[
                status_md,
                current_step_md,
                progress_log_md,
                ranked_df,
                plot_img,
                analysis_json,
                agent_md,
                artifacts_box,
                report_md,
            ],
        )

    return app


def _candidate_ports() -> list[int]:
    explicit = os.environ.get("SITE4DRUG_DEMO_PORT") or os.environ.get("GRADIO_SERVER_PORT")
    if explicit:
        try:
            return [int(explicit)]
        except ValueError:
            pass
    return list(range(7860, 7871))


def launch_app() -> None:
    if not ensure_tinker_api_key(REPO_ROOT):
        raise RuntimeError(
            "TINKER_API_KEY is not set. Run ./scripts/setup_tinker_key.sh and source .tinker.env."
        )
    app = build_app()
    last_error: Exception | None = None
    for port in _candidate_ports():
        try:
            app.queue(default_concurrency_limit=2).launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                show_error=True,
            )
            return
        except OSError as exc:
            last_error = exc
            continue
    raise RuntimeError(
        "Unable to bind Gradio UI to ports 7860-7870. "
        "Set SITE4DRUG_DEMO_PORT to an open port and retry."
    ) from last_error


if __name__ == "__main__":
    launch_app()
