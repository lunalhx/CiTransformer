#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"

RUN_SCRIPT="${PROJECT_ROOT}/scripts/experiments/run_itransformer_experiments.sh"

DATA_DIR="${DATA_DIR:-$(project_config_get paths.data_dir)}"
RESULTS_ROOT="${RESULTS_ROOT:-$(project_config_get paths.results_root)}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$(project_config_get paths.checkpoints_root)}"

REGIME_LABEL_DIR="${REGIME_LABEL_DIR:-${RESULTS_ROOT}/regimes/gmm_hmm_daytime_k7}"
REGIME_GRAPH_ROOT="${REGIME_GRAPH_ROOT:-${RESULTS_ROOT}/causal_graphs/regime_target_pcmci_k7}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${RESULTS_ROOT}/itransformer_regime_dynamic_pcmci_k7}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${CHECKPOINTS_ROOT}/itransformer_regime_dynamic_pcmci_k7}"

PRED_LENS="${PRED_LENS:-1 12 24 48}"
SEQ_LEN="${SEQ_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"
CAUSAL_MASK_MODE="soft_bias"
CAUSAL_MASK_BETA="1.0"
REGIME_MASK_SELECTION="input_end"

export DATA_DIR
export RESULTS_BASE_DIR
export CHECKPOINT_BASE_DIR
export REGIME_LABEL_DIR
export REGIME_GRAPH_ROOT
export PRED_LENS
export SEQ_LEN
export BATCH_SIZE
export LEARNING_RATE
export EPOCHS
export PATIENCE
export CAUSAL_MASK_MODE
export CAUSAL_MASK_BETA
export REGIME_MASK_SELECTION

bash "${RUN_SCRIPT}" "$@"
