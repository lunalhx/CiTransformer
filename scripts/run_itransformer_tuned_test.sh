#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/project_config.sh"
cd "${PROJECT_ROOT}"

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}

export PYTHON_BIN
export MODE="${MODE:-export_tuned_best}"
export REPORT_SPLIT="${REPORT_SPLIT:-test}"
export PRED_LENS="${PRED_LENS:-1 12 24 48}"
export RUN_PRED_LEN1_REF="${RUN_PRED_LEN1_REF:-1}"
export TUNING_PLAN="${TUNING_PLAN:-standard}"
export DATA_DIR="${DATA_DIR:-$(project_config_get paths.data_dir)}"
export RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-$(project_config_get paths.results.itransformer_tuned)}"
export TUNING_RESULTS_ROOT="${TUNING_RESULTS_ROOT:-$(project_config_get paths.results.tuning_itransformer)/${TUNING_PLAN}}"
export TUNING_SUMMARY_ROOT="${TUNING_SUMMARY_ROOT:-${TUNING_RESULTS_ROOT}/summary}"
export DEVICE="${DEVICE:-$(project_config_get runtime.device)}"
export NUM_WORKERS="${NUM_WORKERS:-$(project_config_get runtime.num_workers)}"

echo "======================================================================"
echo "iTransformer tuned test export"
echo "Mode          -> ${MODE}"
echo "Report split  -> ${REPORT_SPLIT}"
echo "Pred lens     -> ${PRED_LENS}"
echo "Data dir      -> $(project_path "${DATA_DIR}")"
echo "Results dir   -> $(project_path "${RESULTS_BASE_DIR}")"
echo "Tuning summary-> $(project_path "${TUNING_SUMMARY_ROOT}")"
echo "======================================================================"

bash "${PROJECT_ROOT}/scripts/run_itransformer_experiments.sh"
