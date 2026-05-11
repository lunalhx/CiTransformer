#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${PROJECT_ROOT}/archives}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_PATH="${ARCHIVE_PATH:-${ARCHIVE_ROOT}/results_csv_${TIMESTAMP}.tar.gz}"
MANIFEST_PATH="${MANIFEST_PATH:-${ARCHIVE_ROOT}/results_csv_${TIMESTAMP}.manifest.txt}"
DELETE_AFTER_ARCHIVE="${DELETE_AFTER_ARCHIVE:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/archive_predictions_csv.sh [options]

Options:
  --archive PATH            Output .tar.gz path.
                            Default: archives/results_csv_<timestamp>.tar.gz
  --manifest PATH           Output manifest path.
                            Default: archives/results_csv_<timestamp>.manifest.txt
  --keep-originals          Keep archived CSV files under results/.
                            By default, archived CSV files are deleted after tar succeeds.
  -h, --help                Show this help.

Environment:
  ARCHIVE_ROOT              Directory for archive outputs. Default: <project>/archives
  ARCHIVE_PATH              Same as --archive.
  MANIFEST_PATH             Same as --manifest.
  DELETE_AFTER_ARCHIVE=0    Same as --keep-originals.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --archive)
      ARCHIVE_PATH="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --keep-originals)
      DELETE_AFTER_ARCHIVE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "${ARCHIVE_PATH}")" "$(dirname "${MANIFEST_PATH}")"

cd "${PROJECT_ROOT}"

find results -type f -name "*.csv" | sort > "${MANIFEST_PATH}"

file_count="$(wc -l < "${MANIFEST_PATH}" | tr -d ' ')"
if [[ "${file_count}" == "0" ]]; then
  echo "No CSV files found under ${PROJECT_ROOT}/results."
  rm -f "${MANIFEST_PATH}"
  exit 0
fi

echo "Found ${file_count} CSV files under results/."
echo "Manifest: ${MANIFEST_PATH}"
echo "Archive : ${ARCHIVE_PATH}"

tar -czf "${ARCHIVE_PATH}" -T "${MANIFEST_PATH}"

echo "Archive created:"
du -h "${ARCHIVE_PATH}"

if [[ "${DELETE_AFTER_ARCHIVE}" == "1" ]]; then
  echo "Deleting archived CSV files listed in manifest..."
  while IFS= read -r path; do
    [[ -n "${path}" ]] && rm -f "${path}"
  done < "${MANIFEST_PATH}"
  echo "Deleted archived CSV files."
else
  echo "Original CSV files were kept."
fi
