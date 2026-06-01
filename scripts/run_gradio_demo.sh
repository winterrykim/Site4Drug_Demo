#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -f ".tinker.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".tinker.env"
  set +a
fi

if [[ -f ".openrouter.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".openrouter.env"
  set +a
fi

python -m site4drug_inference.demo.gradio_demo "$@"
