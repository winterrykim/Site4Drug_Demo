#!/usr/bin/env python3
"""Inline notebook widget demo for Site4Drug."""

from __future__ import annotations

import html
import json
import tempfile
import traceback
from pathlib import Path
from typing import Any

try:
    import ipywidgets as widgets
except ImportError as exc:  # pragma: no cover - notebook-only dependency
    raise RuntimeError("ipywidgets is required for the notebook demo. Install it with `python -m pip install ipywidgets`.") from exc

from IPython.display import Image, Markdown, display

from site4drug_inference.demo.notebook_utils import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    ensure_api_key_or_raise,
    run_notebook_prediction,
)
from site4drug_inference.demo.ui_helpers import (
    agent_conclusion_md,
    build_analysis_json,
    build_artifacts_text,
    load_report_markdown,
    ranked_candidates_display_df,
    resolve_demo_input_sequence,
    resolve_plot_path,
)


def _first_uploaded_file(upload_value: Any) -> dict[str, Any] | None:
    if not upload_value:
        return None
    if isinstance(upload_value, dict):
        entries = list(upload_value.values())
    else:
        entries = list(upload_value)
    if not entries:
        return None
    first = entries[0]
    if isinstance(first, dict):
        return first
    if hasattr(first, "keys"):
        return dict(first)
    return {
        "name": getattr(first, "name", None),
        "content": getattr(first, "content", b""),
    }


def _uploaded_file_to_path(upload_value: Any) -> str | None:
    uploaded = _first_uploaded_file(upload_value)
    if not uploaded:
        return None
    name = str(uploaded.get("name") or "uploaded_sequence.fasta")
    content = uploaded.get("content", b"")
    if isinstance(content, memoryview):
        data = content.tobytes()
    else:
        data = bytes(content)
    temp_dir = Path(tempfile.mkdtemp(prefix="site4drug_upload_"))
    output_path = temp_dir / Path(name).name
    output_path.write_bytes(data)
    return str(output_path)


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _status_html(*, heading: str, items: list[str]) -> str:
    if not items:
        return f"<h3>{_escape(heading)}</h3>"
    bullets = "".join(f"<li>{_escape(item)}</li>" for item in items)
    return f"<h3>{_escape(heading)}</h3><ul>{bullets}</ul>"


def _parse_sampling_seed(raw_value: Any) -> int | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError("Sampling seed must be an integer or blank.") from exc


def build_notebook_app(repo_root: str | Path | None = None) -> widgets.Widget:
    """Build an inline widget app that mirrors the Gradio demo."""
    repo_root = Path(repo_root or Path.cwd()).resolve()

    uniprot_w = widgets.Text(
        value="P29996",
        description="UniProt / Label",
        layout=widgets.Layout(width="50%"),
        style={"description_width": "140px"},
    )
    input_mode_w = widgets.RadioButtons(
        options=["Paste sequence", "Upload FASTA", "UniProt lookup only"],
        value="Paste sequence",
        description="Input mode",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="45%"),
    )
    sequence_text_w = widgets.Textarea(
        value="",
        description="Sequence",
        placeholder="Paste amino-acid sequence or FASTA.",
        layout=widgets.Layout(width="100%", height="160px"),
        style={"description_width": "140px"},
    )
    sequence_upload_w = widgets.FileUpload(
        accept=".fasta,.fa,.txt",
        multiple=False,
        description="Upload FASTA/TXT",
    )
    upload_note_w = widgets.HTML("<small>Use this when Input mode is set to Upload FASTA.</small>")

    mode_w = widgets.Dropdown(options=["auto", "epitope", "pocket"], value="auto", description="Mode")
    top_k_w = widgets.IntSlider(value=5, min=1, max=15, step=1, description="Top-K")
    self_consistency_k_w = widgets.IntSlider(value=1, min=1, max=5, step=1, description="Self-consistency K")
    ptm_source_w = widgets.Dropdown(
        options=["musitedeep", "hybrid", "multi_rule", "glyco_only"],
        value="musitedeep",
        description="PTM source",
    )
    ptm_policy_w = widgets.Dropdown(options=["tiered", "hard", "soft"], value="tiered", description="PTM policy")
    use_motif_w = widgets.Checkbox(value=True, description="Enable motif scan")
    use_base_model_w = widgets.Checkbox(value=False, description="Use base model only")
    base_model_w = widgets.Text(
        value=DEFAULT_BASE_MODEL,
        description="Base model",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "140px"},
    )
    checkpoint_w = widgets.Text(
        value=DEFAULT_CHECKPOINT,
        description="Checkpoint",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "140px"},
    )
    max_tokens_w = widgets.IntSlider(value=3000, min=256, max=12000, step=256, description="Max output tokens")
    max_input_tokens_w = widgets.IntSlider(
        value=10000,
        min=2000,
        max=30000,
        step=500,
        description="Max input tokens",
    )
    sampling_seed_w = widgets.Text(
        value="42",
        description="Sampling seed",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "140px"},
    )
    output_dir_w = widgets.Text(
        value=str(DEFAULT_OUTPUT_DIR),
        description="Output directory",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "140px"},
    )
    use_iedb_validation_w = widgets.Checkbox(value=True, description="Enable IEDB validation")
    no_online_lookup_w = widgets.Checkbox(value=False, description="Disable online UniProt lookup")

    run_btn = widgets.Button(description="Run Site4Drug", button_style="success", icon="play")

    status_w = widgets.HTML(_status_html(heading="Ready", items=["Configure inputs, then run the notebook demo."]))
    current_step_w = widgets.HTML("<b>Current step:</b> idle")
    progress_log_w = widgets.Textarea(
        value="",
        layout=widgets.Layout(width="100%", height="220px"),
        disabled=True,
    )
    analysis_json_w = widgets.Textarea(
        value="{}",
        layout=widgets.Layout(width="100%", height="220px"),
        disabled=True,
    )
    artifacts_w = widgets.Textarea(
        value="",
        layout=widgets.Layout(width="100%", height="140px"),
        disabled=True,
    )
    report_w = widgets.Textarea(
        value="",
        layout=widgets.Layout(width="100%", height="360px"),
        disabled=True,
    )
    ranked_out = widgets.Output()
    plot_out = widgets.Output()
    agent_out = widgets.Output()

    def _clear_results() -> None:
        ranked_out.clear_output()
        plot_out.clear_output()
        agent_out.clear_output()
        analysis_json_w.value = "{}"
        artifacts_w.value = ""
        report_w.value = ""

    def _sync_input_mode(*_: Any) -> None:
        mode_value = str(input_mode_w.value or "").lower()
        paste_visible = mode_value.startswith("paste")
        upload_visible = mode_value.startswith("upload")
        sequence_text_box.layout.display = "" if paste_visible else "none"
        sequence_upload_box.layout.display = "" if upload_visible else "none"

    def _sync_base_model_toggle(*_: Any) -> None:
        checkpoint_w.disabled = bool(use_base_model_w.value)

    def _progress_sink_factory(progress_lines: list[str]):
        def _sink(event: dict[str, Any]) -> None:
            label = str(event.get("label", event.get("step_key", "step")))
            status = str(event.get("status", "running"))
            current_step_w.value = f"<b>Current step:</b> {_escape(label)} [{_escape(status)}]"
            progress_lines.append(
                f"- {event.get('timestamp_utc', '')} {event.get('event_type', '')} {label} -> {status}"
            )
            progress_log_w.value = "\n".join(progress_lines[-60:]) or "- Waiting for events..."

        return _sink

    def _run(_: widgets.Button) -> None:
        progress_lines: list[str] = []
        run_btn.disabled = True
        _clear_results()
        status_w.value = _status_html(heading="Run in progress", items=["Preparing inputs and starting inference."])
        current_step_w.value = "<b>Current step:</b> Initializing run"
        progress_log_w.value = ""

        try:
            ensure_api_key_or_raise(repo_root)
            uploaded_path = _uploaded_file_to_path(sequence_upload_w.value)
            uniprot = str(uniprot_w.value or "UNKNOWN").strip() or "UNKNOWN"
            raw_sequence, input_source = resolve_demo_input_sequence(
                uniprot=uniprot,
                input_mode=input_mode_w.value,
                sequence_text=sequence_text_w.value,
                sequence_file=uploaded_path,
                allow_online_lookup=not bool(no_online_lookup_w.value),
                paste_source_label="notebook_sequence_text",
                upload_source_prefix="notebook_upload:",
            )
            sampling_seed = _parse_sampling_seed(sampling_seed_w.value)
            result = run_notebook_prediction(
                uniprot=uniprot,
                raw_sequence=raw_sequence,
                checkpoint=str(checkpoint_w.value or DEFAULT_CHECKPOINT),
                base_model=str(base_model_w.value or DEFAULT_BASE_MODEL),
                use_base_model=bool(use_base_model_w.value),
                mode=str(mode_w.value),
                candidate_source="llm_propose",
                top_k=int(top_k_w.value),
                max_tokens=int(max_tokens_w.value),
                self_consistency_k=int(self_consistency_k_w.value),
                sampling_seed=sampling_seed,
                max_input_tokens=int(max_input_tokens_w.value),
                output_dir=Path(output_dir_w.value or str(DEFAULT_OUTPUT_DIR)).expanduser(),
                enable_plot=True,
                use_multi_agent=True,
                input_source=input_source,
                repair_with_base_model=True,
                panel_with_base_model=True,
                ptm_source=str(ptm_source_w.value),
                ptm_policy=str(ptm_policy_w.value),
                motif_source="remote",
                use_motif=bool(use_motif_w.value),
                use_iedb_validation=bool(use_iedb_validation_w.value),
                show_progress=False,
                progress_sink=_progress_sink_factory(progress_lines),
            )
            payload = result.get("run_payload", {}) or {}
            generation = payload.get("generation", {}) or {}
            status_w.value = _status_html(
                heading="Run complete",
                items=[
                    f"Run status: {payload.get('run_status', 'unknown')}",
                    (
                        "Recommended modality: "
                        f"{payload.get('recommended_modality', 'unknown')} "
                        f"(confidence {float(payload.get('modality_confidence', 0.0) or 0.0):.2f})"
                    ),
                    f"Token strategy: {payload.get('token_strategy_used', 'unknown')}",
                    f"Candidate source: {generation.get('candidate_source', 'unknown')}",
                    f"Sampling seed: {generation.get('sampling_seed', None)}",
                    f"Parsed candidates: {int(payload.get('parsed_candidate_count', 0) or 0)}",
                    f"Output directory: {result.get('run_dir')}",
                ],
            )
            current_step_w.value = "<b>Current step:</b> Completed"
            progress_log_w.value = "\n".join(progress_lines) or "- No progress events captured."

            with ranked_out:
                ranked_out.clear_output()
                display(ranked_candidates_display_df(payload))

            plot_path = resolve_plot_path(result, payload)
            with plot_out:
                plot_out.clear_output()
                if plot_path:
                    display(Image(filename=plot_path))
                else:
                    print("No plot artifact generated.")

            analysis_json_w.value = json.dumps(build_analysis_json(payload), indent=2)
            with agent_out:
                agent_out.clear_output()
                display(Markdown(agent_conclusion_md(payload)))
            artifacts_w.value = build_artifacts_text(result, payload)
            report_w.value = load_report_markdown(result)
        except Exception as exc:
            status_w.value = _status_html(heading="Run failed", items=[str(exc)])
            current_step_w.value = "<b>Current step:</b> Failed"
            trace = traceback.format_exc()
            progress_log_w.value = ("\n".join(progress_lines) + "\n\n" + trace).strip()
            with ranked_out:
                ranked_out.clear_output()
                display(ranked_candidates_display_df({}))
            with plot_out:
                plot_out.clear_output()
            analysis_json_w.value = json.dumps({"error": str(exc)}, indent=2)
            with agent_out:
                agent_out.clear_output()
                display(Markdown("- Run failed before agent conclusions could be produced."))
            artifacts_w.value = ""
            report_w.value = ""
        finally:
            run_btn.disabled = False

    input_controls = widgets.HBox([uniprot_w, input_mode_w])
    sequence_text_box = widgets.VBox([sequence_text_w])
    sequence_upload_box = widgets.VBox([sequence_upload_w, upload_note_w])

    control_rows = widgets.VBox(
        [
            widgets.HBox([mode_w, top_k_w, self_consistency_k_w]),
            widgets.HBox([ptm_source_w, ptm_policy_w]),
            widgets.HBox([use_motif_w, use_base_model_w]),
            widgets.VBox([base_model_w, checkpoint_w]),
            widgets.HBox([max_tokens_w, max_input_tokens_w]),
            widgets.VBox([sampling_seed_w]),
            widgets.VBox([output_dir_w]),
            widgets.HBox([use_iedb_validation_w, no_online_lookup_w]),
        ]
    )
    controls_accordion = widgets.Accordion(children=[control_rows], selected_index=0)
    controls_accordion.set_title(0, "Inference controls")

    summary_panel = widgets.VBox(
        [
            status_w,
            current_step_w,
            widgets.HTML("<b>Progress log</b>"),
            progress_log_w,
        ]
    )
    ranked_panel = widgets.VBox([ranked_out])
    plot_panel = widgets.VBox([plot_out])
    analysis_panel = widgets.VBox([analysis_json_w])
    agent_panel = widgets.VBox([agent_out])
    artifacts_panel = widgets.VBox([artifacts_w])
    report_panel = widgets.VBox([report_w])

    results_tabs = widgets.Tab(
        children=[
            summary_panel,
            ranked_panel,
            plot_panel,
            analysis_panel,
            agent_panel,
            artifacts_panel,
            report_panel,
        ]
    )
    for idx, title in enumerate(
        ["Status", "Ranked candidates", "Plot", "Analysis JSON", "Agent conclusions", "Artifacts", "Report markdown"]
    ):
        results_tabs.set_title(idx, title)

    use_base_model_w.observe(_sync_base_model_toggle, names="value")
    input_mode_w.observe(_sync_input_mode, names="value")
    run_btn.on_click(_run)
    _sync_base_model_toggle()
    _sync_input_mode()

    return widgets.VBox(
        [
            widgets.HTML(
                "<h2>Site4Drug Notebook Demo</h2>"
                "<p>Inline notebook UI with the same inputs and outputs as the Gradio demo, without needing a local port.</p>"
            ),
            input_controls,
            sequence_text_box,
            sequence_upload_box,
            controls_accordion,
            run_btn,
            results_tabs,
        ]
    )


__all__ = ["build_notebook_app"]
