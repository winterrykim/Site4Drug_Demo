# Site4Drug Inference

Inference-only distribution repo for Site4Drug. This repository keeps the successful default inference path, compact reports, reproducibility notebooks, and optional Gradio demo, while leaving training and local-only benchmarking/evaluation scripts out of the committed tree.

## Included
- Packaged inference runtime under `site4drug_inference/`
- `predict` CLI
- Compact Markdown/HTML report generation with the same current style/output
- Optional `demo` CLI for the Gradio app
- Reproducibility notebooks under `notebooks/demo/`
- Minimal reference data bundle under `data/`

## Not Included
- Training code
- Training-data prep
- Committed evaluation/benchmark scripts
- Output artifacts
- Bulky benchmark assets

## Installation
```bash
python -m pip install -e .
```

Optional demo dependencies:
```bash
python -m pip install -e .[demo]
```

Optional notebook dependencies:
```bash
python -m pip install -e .[notebooks]
```

## CLI
Run a prediction:
```bash
predict --uniprot P29996 --sequence-file antigen.fasta --mode auto --top-k 5
```

Launch the optional demo:
```bash
demo
```

## Defaults Preserved
- `auto` mode
- `llm_propose` candidate generation
- `musitedeep` PTM source
- `tiered` PTM policy
- remote motif scan
- multi-agent panel enabled
- `self_consistency_k=1` default
- current `self_consistency_k=3` vote/cluster aggregation logic
- current compact Markdown/HTML report style

## Local-Only `.local/` Area
Local evaluation, benchmarking, and one-off comparison scripts belong in `.local/`.

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
