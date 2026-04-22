#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Edit these if your environment is different.
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
PLAN="${PLAN:-standard}"
MAIN_PRED_LENS="${MAIN_PRED_LENS:-12 24}"
RUN_PRED_LEN1_REF="${RUN_PRED_LEN1_REF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Training defaults for the formal tuning round.
SEQ_LEN="${SEQ_LEN:-96}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
MIN_DELTA="${MIN_DELTA:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"
PROGRESS_MININTERVAL="${PROGRESS_MININTERVAL:-15}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export PYTHON_BIN
export DEVICE
export PLAN
export MAIN_PRED_LENS
export RUN_PRED_LEN1_REF
export SKIP_EXISTING
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
