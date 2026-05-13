#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/train/run_lstm.py"

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
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
MIN_DELTA="${MIN_DELTA:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-$(project_config_get runtime.num_workers)}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"
PROGRESS_MININTERVAL="${PROGRESS_MININTERVAL:-15}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-$(project_config_get runtime.device)}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-$(project_config_get paths.results.lstm)}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$(project_config_get paths.checkpoints.lstm)}"
TIME_COL="${TIME_COL:-}"
SAMPLING_FREQ_MINUTES="${SAMPLING_FREQ_MINUTES:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
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
  local checkpoint_path="${CHECKPOINT_BASE_DIR}/pred_len_${pred_len}/best_model.pth"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${RUN_SCRIPT}"
    --data_dir "${DATA_DIR}"
    --seq_len "${SEQ_LEN}"
    --pred_len "${pred_len}"
    --batch_size "${BATCH_SIZE}"
    --hidden_size "${HIDDEN_SIZE}"
    --num_layers "${NUM_LAYERS}"
    --dropout "${DROPOUT}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_clip "${GRAD_CLIP}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --min_delta "${MIN_DELTA}"
    --num_workers "${NUM_WORKERS}"
    --log_interval "${LOG_INTERVAL}"
    --progress_mininterval "${PROGRESS_MININTERVAL}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --results_dir "${results_dir}"
    --checkpoint_path "${checkpoint_path}"
  )

  if [[ -n "${TIME_COL}" ]]; then
    cmd+=(--time_col "${TIME_COL}")
  fi

  if [[ -n "${SAMPLING_FREQ_MINUTES}" ]]; then
    cmd+=(--sampling_freq_minutes "${SAMPLING_FREQ_MINUTES}")
  fi

  if [[ -n "${MAX_TRAIN_BATCHES}" ]]; then
    cmd+=(--max_train_batches "${MAX_TRAIN_BATCHES}")
  fi

  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    cmd+=(--max_eval_batches "${MAX_EVAL_BATCHES}")
  fi

  if [[ "$#" -gt 1 ]]; then
    cmd+=("${@:2}")
  fi

  echo "======================================================================"
  echo "Running LSTM baseline with pred_len=${pred_len}"
  echo "Project    -> ${PROJECT_ROOT}"
  echo "Python     -> ${PYTHON_BIN}"
  echo "Data dir   -> $(project_path "${DATA_DIR}")"
  echo "Results    -> $(project_path "${results_dir}")"
  echo "Checkpoint -> $(project_path "${checkpoint_path}")"
  echo "======================================================================"

  "${cmd[@]}"
}

for pred_len in "${PRED_LEN_ARRAY[@]}"; do
  run_case "${pred_len}" "$@"
done

echo "All LSTM experiments finished."
