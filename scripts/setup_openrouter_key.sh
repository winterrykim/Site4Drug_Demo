#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.openrouter.env}"

printf "OpenRouter API key (input hidden): "
stty -echo
read -r OPENROUTER_API_KEY
stty echo
printf "\n"

if [[ -z "${OPENROUTER_API_KEY}" ]]; then
  echo "No key provided; aborting." >&2
  exit 1
fi

printf "OpenRouter model id (optional; Enter to use account default): "
read -r OPENROUTER_MODEL

cat > "${ENV_FILE}" <<EOF
export OPENROUTER_API_KEY='${OPENROUTER_API_KEY}'
export OPENROUTER_MODEL='${OPENROUTER_MODEL}'
export OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'
export OPENROUTER_TITLE='Site4Drug Demo'
EOF

chmod 600 "${ENV_FILE}"
echo "Wrote ${ENV_FILE}. Run: source ${ENV_FILE}"
