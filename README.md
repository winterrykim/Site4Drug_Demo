# Site4Drug Inference

Inference-only distribution repo for Site4Drug. This repository keeps the successful default inference path, compact reports, reproducibility notebooks, and an optional Gradio demo, while leaving training and local-only benchmarking/evaluation scripts out of the committed tree.

## What This Repo Includes
- Packaged inference runtime under `site4drug_inference/`
- `predict` CLI for batchable inference runs
- Compact Markdown and HTML report generation with the current Site4Drug report style
- Plot artifact generation by default
- Optional `demo` CLI for the Gradio app
- Reproducibility notebooks under `notebooks/demo/`
- Minimal reference data bundle under `data/`

## What This Repo Does Not Include
- Training code
- Training-data prep
- Committed evaluation or benchmarking scripts
- Output artifacts
- Bulky benchmark assets

## Setup
Full setup for CLI, demo, and notebooks:

```bash
cd <repo-root>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[demo,notebooks]
./scripts/setup_tinker_key.sh
source .tinker.env
```

`./scripts/setup_tinker_key.sh` prompts you for `TINKER_API_KEY` and writes it to `.tinker.env` with restricted permissions.

If you only need CLI inference:

```bash
python -m pip install -e .
./scripts/setup_tinker_key.sh
source .tinker.env
```

Important:

- The demo requires `TINKER_API_KEY`.
- CLI inference requires `TINKER_API_KEY`.
- Notebook inference requires `TINKER_API_KEY`.
- `--use-base-model` only skips the fine-tuned checkpoint. It still uses Tinker for base-model inference.

Optional environment variables:

- `SITE4DRUG_OUTPUT_DIR`
  Sets the default output root. If unset, predictions are written under `outputs/predictions` inside this repo.
- `SITE4DRUG_MUSITEDEEP_API_BASE_URL`
  Overrides the default MusiteDeep API base URL.

## Run
CLI help:

```bash
predict --help
```

Run with a raw sequence:

```bash
predict \
  --uniprot TEST_SEQ \
  --sequence ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY \
  --mode auto \
  --top-k 5
```

Run from a FASTA file:

```bash
predict \
  --uniprot P29996 \
  --sequence-file antigen.fasta \
  --mode auto \
  --top-k 5
```

Launch the demo:

```bash
./scripts/run_gradio_demo.sh
```

The launcher activates `.venv` if present, loads `.tinker.env` if present, and runs `python -m site4drug_inference.demo.gradio_demo`.

To force a demo port:

```bash
SITE4DRUG_DEMO_PORT=7890 ./scripts/run_gradio_demo.sh
```

## How Sequence Input Works
`predict` resolves the input sequence in this order:

1. `--sequence-file`
   Reads a FASTA file or plain-text sequence file. This takes precedence if multiple input modes are supplied.
2. `--interactive`
   Prompts for the label and sequence in the terminal.
3. `--sequence`
   Takes the raw sequence directly from the CLI.
4. `--uniprot`
   If no sequence was provided, the CLI attempts sequence resolution from the UniProt label/accession unless `--no-online-lookup` is set.

Notes:

- `--sequence-file` accepts either FASTA or plain text.
- If the FASTA header is present and `--uniprot` is still `UNKNOWN`, the first token of the FASTA header is used as the run label.
- Raw sequences are normalized before inference: whitespace is removed, residues are uppercased, and terminal `*` characters are stripped.

## Default Behavior
The committed repo keeps the current successful default path:

- `auto` mode
- `llm_propose` candidate generation
- `musitedeep` PTM source
- `tiered` PTM policy
- remote motif scan
- multi-agent panel enabled
- `self_consistency_k=1` by default
- current `self_consistency_k=3` vote and cluster aggregation logic when enabled
- current compact Markdown and HTML report style

## What Gets Written By Default
Yes. A prediction run generates report artifacts by default.

Unless you override `--output-dir`, each run is written under:

```text
outputs/predictions/<timestamp>_<label>/
```

Typical artifact bundle:

- `prediction_log.json`
  Full structured run payload and provenance
- `prediction_report.md`
  Compact Markdown report
- `prediction_report.html`
  Compact HTML report
- `hydropathy_ptm_plot.png`
  Hydropathy/PTM/candidate visualization, unless `--no-plot` is used
- `hydropathy_ptm_plot.json`
  Structured payload describing the plot inputs, unless `--no-plot` is used
- `agent_traces.json`
  Multi-agent panel traces
- `self_consistency/`
  Per-attempt artifacts when `--self-consistency-k` is greater than 1

At the end of a CLI run, the console prints the paths to:

- the JSON log
- the Markdown report
- the HTML report

## Key Runtime Options
- `--mode {auto,epitope,pocket}`
  Requested inference mode.
- `--top-k`
  Requested number of ranked candidates. Runs can return fewer rows now that deterministic top-k backfill is removed.
- `--self-consistency-k`
  Repeats proposal generation and aggregates overlapping candidates.
- `--sampling-seed`
  Base seed for reproducible sampling when supported by the backend.
- `--ptm-source {musitedeep,hybrid,glyco_only,multi_rule}`
  PTM feature source.
- `--ptm-policy {tiered,hard,soft}`
  PTM penalty policy.
- `--no-motif`
  Disables motif lookup.
- `--no-iedb-validation`
  Disables optional IEDB validation.
- `--no-online-lookup`
  Prevents fallback sequence resolution from UniProt when no sequence is supplied.
- `--no-plot`
  Disables plot artifact generation.

## Optional Gradio Demo
Install the demo extras first:

```bash
python -m pip install -e .[demo]
```

Then launch:

```bash
demo
```

The demo uses the same committed inference pipeline and produces the same artifact/report outputs as the CLI.

## Local-Only `.local/` Area
Local evaluation, benchmarking, and one-off comparison scripts belong in `.local/`. That folder is intentionally gitignored.

Suggested layout:

- `.local/eval/`
- `.local/benchmarks/`
- `.local/scripts/`
- `.local/notebooks/`

The committed inference runtime does not depend on `.local/`.

## Shipped Data
- `data/Site4Drug_GroundTruth.json`
- `data/Site4Drug_GroundTruth_newest_0324.json`
- `data/TCellEpitope_GroundTruth_random100_seed42.json`
- `data/TCellEpitope_GroundTruth_random100_seed42_audit.csv`
- `data/combined/tcell_regions_with_seq.parquet`

## Repo Layout
```text
site4drug_inference/
  common/   # feature extraction, PTM/motif integration, schemas, sampling helpers
  demo/     # predict CLI, report rendering, plotting, panel/orchestrator logic, Gradio demo
data/       # shipped reference data
notebooks/  # reproducibility notebooks
tests/      # regression and smoke-style tests for the committed inference path
```
