#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"

RUN_SCRIPT="${PROJECT_ROOT}/scripts/experiments/run_itransformer_experiments.sh"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Error: cannot find ${RUN_SCRIPT}" >&2
  exit 1
fi

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}

DATA_DIR="${DATA_DIR:-$(project_config_get paths.data_dir)}"
RESULTS_ROOT="${RESULTS_ROOT:-$(project_config_get paths.results_root)}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$(project_config_get paths.checkpoints_root)}"
CAUSAL_GRAPH_DIR="${CAUSAL_GRAPH_DIR:-$(project_config_get paths.results.causal_graphs_global_pcmci_11vars_train)}"
PRED_LENS="${PRED_LENS:-12 24 48}"
RUN_NONE="${RUN_NONE:-1}"
RUN_SOFT_BETA_1="${RUN_SOFT_BETA_1:-1}"
RUN_SOFT_BETA_2="${RUN_SOFT_BETA_2:-1}"
RUN_HARD="${RUN_HARD:-0}"

SEQ_LEN="${SEQ_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"

cd "${PROJECT_ROOT}"

CAUSAL_GRAPH_PATH="$(project_path "${CAUSAL_GRAPH_DIR}")"
if [[ ! -f "${CAUSAL_GRAPH_PATH}/global_causal_adjacency.csv" ]]; then
  echo "Error: cannot find current global PCMCI adjacency under ${CAUSAL_GRAPH_PATH}" >&2
  echo "Run bash scripts/run_experiment.sh global-pcmci-mask once, or set CAUSAL_GRAPH_DIR." >&2
  exit 1
fi

run_case() {
  local label="$1"
  local mask_mode="$2"
  local mask_beta="$3"
  shift 3

  local results_base_dir="${RESULTS_ROOT}/itransformer_mask_calibration/${label}"
  local checkpoint_base_dir="${CHECKPOINTS_ROOT}/itransformer_mask_calibration/${label}"

  echo "======================================================================"
  echo "Running iTransformer mask calibration case: ${label}"
  echo "Mask mode    -> ${mask_mode}"
  echo "Mask beta    -> ${mask_beta}"
  echo "Pred lens    -> ${PRED_LENS}"
  echo "Results      -> $(project_path "${results_base_dir}")"
  echo "Checkpoints  -> $(project_path "${checkpoint_base_dir}")"
  echo "======================================================================"

  PYTHON_BIN="${PYTHON_BIN}" \
  DATA_DIR="${DATA_DIR}" \
  MODE="train" \
  SEQ_LEN="${SEQ_LEN}" \
  PRED_LENS="${PRED_LENS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  EPOCHS="${EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  NUM_WORKERS="${NUM_WORKERS:-$(project_config_get runtime.num_workers)}" \
  DEVICE="${DEVICE:-$(project_config_get runtime.device)}" \
  RESULTS_BASE_DIR="${results_base_dir}" \
  CHECKPOINT_BASE_DIR="${checkpoint_base_dir}" \
  CAUSAL_MASK_MODE="${mask_mode}" \
  CAUSAL_MASK_BETA="${mask_beta}" \
  bash "${RUN_SCRIPT}" "$@"
}

if [[ "${RUN_NONE}" == "1" ]]; then
  run_case "none_matched" "none" "1.0"
fi

if [[ "${RUN_SOFT_BETA_1}" == "1" ]]; then
  run_case "soft_bias_beta_1" "soft_bias" "1.0" --causal_graph_dir "${CAUSAL_GRAPH_DIR}"
fi

if [[ "${RUN_SOFT_BETA_2}" == "1" ]]; then
  run_case "soft_bias_beta_2" "soft_bias" "2.0" --causal_graph_dir "${CAUSAL_GRAPH_DIR}"
fi

if [[ "${RUN_HARD}" == "1" ]]; then
  run_case "hard_matched" "hard" "1.0" --causal_graph_dir "${CAUSAL_GRAPH_DIR}"
fi

echo "All requested iTransformer mask calibration cases finished."
