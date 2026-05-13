#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ops/restore_predictions_csv.sh <archive.tar.gz>

Restores archived results CSV files back into their original results/... paths.
The archive must have been created by scripts/ops/archive_predictions_csv.sh.
EOF
}

if [[ "$#" -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ARCHIVE_PATH="$1"
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Error: archive not found: ${ARCHIVE_PATH}" >&2
  exit 1
fi

echo "Restoring ${ARCHIVE_PATH}"
echo "Project root: ${PROJECT_ROOT}"

tar -xzf "${ARCHIVE_PATH}" -C "${PROJECT_ROOT}"

echo "Restore complete."
