#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_itransformer_experiments.sh"
CAUSAL_SCRIPT="${PROJECT_ROOT}/causal_algo/run_global_pcmci.py"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Error: cannot find ${RUN_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${CAUSAL_SCRIPT}" ]]; then
  echo "Error: cannot find ${CAUSAL_SCRIPT}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-data/processed_long_no_wind_2015_2022}"
CAUSAL_GRAPH_DIR="${CAUSAL_GRAPH_DIR:-results/d1_long_no_wind_2015_2022/causal_graphs/global_pcmci_11vars_train}"
CAUSAL_TRAIN_PATH="${CAUSAL_TRAIN_PATH:-${DATA_DIR}/splits/train.csv}"
REBUILD_CAUSAL_GRAPH="${REBUILD_CAUSAL_GRAPH:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${REBUILD_CAUSAL_GRAPH}" == "1" || ! -f "${CAUSAL_GRAPH_DIR}/global_causal_adjacency.csv" ]]; then
  "${PYTHON_BIN}" -u "${CAUSAL_SCRIPT}" \
    --train_path "${CAUSAL_TRAIN_PATH}" \
    --sample_scope full_train \
    --output_dir "${CAUSAL_GRAPH_DIR}"
fi

PYTHON_BIN="${PYTHON_BIN}" \
MODE="${MODE:-train}" \
DATA_DIR="${DATA_DIR}" \
SEQ_LEN="${SEQ_LEN:-96}" \
PRED_LENS="${PRED_LENS:-1 12 24 48}" \
BATCH_SIZE="${BATCH_SIZE:-128}" \
D_MODEL="${D_MODEL:-192}" \
N_HEADS="${N_HEADS:-4}" \
E_LAYERS="${E_LAYERS:-2}" \
D_FF="${D_FF:-384}" \
FACTOR="${FACTOR:-5}" \
DROPOUT="${DROPOUT:-0.1}" \
ACTIVATION="${ACTIVATION:-gelu}" \
LEARNING_RATE="${LEARNING_RATE:-1e-3}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}" \
GRAD_CLIP="${GRAD_CLIP:-1.0}" \
EPOCHS="${EPOCHS:-120}" \
PATIENCE="${PATIENCE:-20}" \
MIN_DELTA="${MIN_DELTA:-1e-5}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
SEED="${SEED:-42}" \
DEVICE="${DEVICE:-auto}" \
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-results/d1_long_no_wind_2015_2022/itransformer_global_pcmci_11vars}" \
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-checkpoints/d1_long_no_wind_2015_2022/itransformer_global_pcmci_11vars}" \
bash "${RUN_SCRIPT}" --causal_graph_dir "${CAUSAL_GRAPH_DIR}"
