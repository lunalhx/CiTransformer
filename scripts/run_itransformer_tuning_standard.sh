#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/project_config.sh"
cd "${PROJECT_ROOT}"

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}
DEVICE="${DEVICE:-$(project_config_get runtime.device)}"
PLAN="${PLAN:-standard}"
MAIN_PRED_LENS="${MAIN_PRED_LENS:-12 24 48}"
RUN_PRED_LEN1_REF="${RUN_PRED_LEN1_REF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DATA_DIR="${DATA_DIR:-$(project_config_get paths.data_dir)}"

# Training defaults for the formal tuning round.
SEQ_LEN="${SEQ_LEN:-96}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
MIN_DELTA="${MIN_DELTA:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"
PROGRESS_MININTERVAL="${PROGRESS_MININTERVAL:-15}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-$(project_config_get runtime.num_workers)}"

export PYTHON_BIN
export DEVICE
export PLAN
export MAIN_PRED_LENS
export RUN_PRED_LEN1_REF
export SKIP_EXISTING
export DATA_DIR
export SEQ_LEN
export EPOCHS
export PATIENCE
export MIN_DELTA
export GRAD_CLIP
export LOG_INTERVAL
export PROGRESS_MININTERVAL
export SEED
export NUM_WORKERS

bash "${PROJECT_ROOT}/scripts/tune_itransformer.sh"
