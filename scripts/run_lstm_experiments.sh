#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-causal_env}"
SEQ_LEN="${SEQ_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

# Use a writable Matplotlib cache directory to avoid font cache warnings.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-causal_env}"

run_case() {
  local pred_len="$1"
  local results_dir="results/lstm/pred_len_${pred_len}"
  local checkpoint_path="checkpoints/lstm/pred_len_${pred_len}/best_model.pth"

  echo "======================================================================"
  echo "Running LSTM baseline with pred_len=${pred_len}"
  echo "Results   -> ${PROJECT_ROOT}/${results_dir}"
  echo "Checkpoint-> ${PROJECT_ROOT}/${checkpoint_path}"
  echo "======================================================================"

  conda run --no-capture-output -n "${CONDA_ENV}" python -u "${PROJECT_ROOT}/scripts/run_lstm.py" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${pred_len}" \
    --batch_size "${BATCH_SIZE}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_layers "${NUM_LAYERS}" \
    --dropout "${DROPOUT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --log_interval 200 \
    --results_dir "${results_dir}" \
    --checkpoint_path "${checkpoint_path}"
}

run_case 1
run_case 12
run_case 24

echo "All LSTM experiments finished."
