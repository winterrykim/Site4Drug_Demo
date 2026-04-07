#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.tinker.env}"

read -r -s -p "Paste TINKER_API_KEY: " TINKER_API_KEY
echo

if [[ -z "${TINKER_API_KEY}" ]]; then
  echo "No key entered. Aborting."
  exit 1
fi

if [[ "${TINKER_API_KEY}" != tml-* ]]; then
  echo "Warning: key does not start with 'tml-'."
fi

umask 077
printf 'export TINKER_API_KEY=%q\n' "${TINKER_API_KEY}" > "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

echo "Saved ${ENV_FILE} with restricted permissions."
echo "Run: source ${ENV_FILE}"
