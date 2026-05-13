#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/train/run_persistence.py"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Error: cannot find ${RUN_SCRIPT}" >&2
  exit 1
fi

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}

if ! "${PYTHON_BIN}" -c "import torch, pandas, matplotlib" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} is missing required packages (torch/pandas/matplotlib)." >&2
  echo "Hint: current machine prefers ${PROJECT_ROOT}/.venv/bin/python after installing requirements." >&2
  exit 1
fi

setup_matplotlib_cache

DATA_DIR="${DATA_DIR:-$(project_config_get paths.data_dir)}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-1 12 24 48}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-$(project_config_get runtime.num_workers)}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-$(project_config_get runtime.device)}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-$(project_config_get paths.results.persistence)}"
TIME_COL="${TIME_COL:-}"
SAMPLING_FREQ_MINUTES="${SAMPLING_FREQ_MINUTES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"

read -r -a PRED_LEN_ARRAY <<< "${PRED_LENS//,/ }"

if [[ "${#PRED_LEN_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: PRED_LENS is empty." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

run_case() {
  local pred_len="$1"
  local results_dir="${RESULTS_BASE_DIR}/pred_len_${pred_len}"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${RUN_SCRIPT}"
    --data_dir "${DATA_DIR}"
    --seq_len "${SEQ_LEN}"
    --pred_len "${pred_len}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --results_dir "${results_dir}"
  )

  if [[ -n "${TIME_COL}" ]]; then
    cmd+=(--time_col "${TIME_COL}")
  fi

  if [[ -n "${SAMPLING_FREQ_MINUTES}" ]]; then
    cmd+=(--sampling_freq_minutes "${SAMPLING_FREQ_MINUTES}")
  fi

  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    cmd+=(--max_eval_batches "${MAX_EVAL_BATCHES}")
  fi

  if [[ "$#" -gt 1 ]]; then
    cmd+=("${@:2}")
  fi

  echo "======================================================================"
  echo "Running persistence baseline with pred_len=${pred_len}"
  echo "Project -> ${PROJECT_ROOT}"
  echo "Python  -> ${PYTHON_BIN}"
  echo "Data    -> $(project_path "${DATA_DIR}")"
  echo "Results -> $(project_path "${results_dir}")"
  echo "======================================================================"

  "${cmd[@]}"
}

for pred_len in "${PRED_LEN_ARRAY[@]}"; do
  run_case "${pred_len}" "$@"
done

echo "All persistence experiments finished."
