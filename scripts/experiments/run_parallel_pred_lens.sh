#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}

config_get_default() {
  local key="$1"
  local default="$2"

  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" - "${key}" "${default}" <<'PY'
import sys

from utils.project_config import load_project_config

key = sys.argv[1]
default = sys.argv[2]
config = load_project_config()
value = config.get(key, default)

if value is None:
    print(default)
elif isinstance(value, list):
    print(" ".join(str(item) for item in value))
else:
    print(value)
PY
)
}

CONFIG_EXPERIMENT_SCRIPT="$(config_get_default parallel.experiment_script "")"
if [[ -z "${CONFIG_EXPERIMENT_SCRIPT}" ]]; then
  CONFIG_EXPERIMENT_SCRIPT="$(config_get_default parallel.experiment "")"
fi

EXPERIMENT="${EXPERIMENT:-${CONFIG_EXPERIMENT_SCRIPT:-itransformer}}"
PRED_LENS="${PRED_LENS:-$(config_get_default parallel.pred_lens "1 12 24 48")}"
MAX_PARALLEL="${MAX_PARALLEL:-$(config_get_default parallel.max_parallel "")}"
LOG_DIR="${LOG_DIR:-$(config_get_default parallel.log_dir "")}"
COMPUTE_THREADS="${COMPUTE_THREADS:-$(config_get_default parallel.compute_threads "")}"
CASE_MATRIX="${CASE_MATRIX:-$(config_get_default parallel.case_matrix "")}"
DEVICE_OVERRIDE="${DEVICE:-}"
NUM_WORKERS_OVERRIDE="${NUM_WORKERS:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment.sh parallel-pred-lens [options]

Options:
  --experiment NAME       Override local.yaml parallel.experiment_script
                          Supports: itransformer | lstm | persistence | global_pcmci | script path
  --pred-lens LIST        Override local.yaml parallel.pred_lens, e.g. "1 12 24 48" or "1,12,24,48"
  --max-parallel N        Override local.yaml parallel.max_parallel
                          Default: number of pred_lens
  --log-dir DIR           Override local.yaml parallel.log_dir
                          Default: logs/parallel_<experiment>
  --compute-threads N     Override local.yaml parallel.compute_threads
  --case-matrix LIST      Optional cases to run for every pred_len, e.g. "none,soft1,soft2"
                          Supported cases for mask calibration: none | soft1 | soft2 | hard
  --device DEVICE         Optional override: cpu | cuda | auto
  --num-workers N         Optional override for DataLoader workers
  -h, --help              Show this help

Examples:
  bash scripts/run_experiment.sh parallel-pred-lens
  bash scripts/run_experiment.sh parallel-pred-lens --experiment scripts/experiments/run_lstm_experiments.sh --pred-lens "1,12,24,48"
  bash scripts/run_experiment.sh parallel-pred-lens --experiment persistence --pred-lens "1 12"

Notes:
  - By default, experiment_script/pred_lens/device/num_workers/python_bin come from configs/local.yaml.
  - Use command-line options only when you want to temporarily override local.yaml.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --pred-lens)
      PRED_LENS="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --compute-threads)
      COMPUTE_THREADS="$2"
      shift 2
      ;;
    --case-matrix)
      CASE_MATRIX="$2"
      shift 2
      ;;
    --device)
      DEVICE_OVERRIDE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${EXPERIMENT}" in
  itransformer)
    EXPERIMENT_SCRIPT="${PROJECT_ROOT}/scripts/experiments/run_itransformer_experiments.sh"
    EXPERIMENT_LABEL="itransformer"
    ;;
  lstm)
    EXPERIMENT_SCRIPT="${PROJECT_ROOT}/scripts/experiments/run_lstm_experiments.sh"
    EXPERIMENT_LABEL="lstm"
    ;;
  persistence)
    EXPERIMENT_SCRIPT="${PROJECT_ROOT}/scripts/experiments/run_persistence_experiments.sh"
    EXPERIMENT_LABEL="persistence"
    ;;
  global_pcmci|global_pcmci_itransformer)
    EXPERIMENT_SCRIPT="${PROJECT_ROOT}/scripts/experiments/run_global_pcmci_itransformer_11vars.sh"
    EXPERIMENT_LABEL="global_pcmci_itransformer"
    ;;
  *)
    if [[ "${EXPERIMENT}" == /* ]]; then
      EXPERIMENT_SCRIPT="${EXPERIMENT}"
    else
      EXPERIMENT_SCRIPT="${PROJECT_ROOT}/${EXPERIMENT}"
    fi
    EXPERIMENT_LABEL="$(basename "${EXPERIMENT_SCRIPT}" .sh)"
    ;;
esac

if [[ ! -f "${EXPERIMENT_SCRIPT}" ]]; then
  echo "Error: experiment script does not exist: ${EXPERIMENT_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${COMPUTE_THREADS}" ]] && ! [[ "${COMPUTE_THREADS}" =~ ^[0-9]+$ ]]; then
  echo "Error: --compute-threads must be a positive integer." >&2
  exit 1
fi

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${PROJECT_ROOT}/logs/parallel_${EXPERIMENT_LABEL}"
elif [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${PROJECT_ROOT}/${LOG_DIR}"
fi

mkdir -p "${LOG_DIR}"

read -r -a PRED_LEN_ARRAY <<< "${PRED_LENS//,/ }"
if [[ "${#PRED_LEN_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: --pred-lens is empty." >&2
  exit 1
fi

if [[ -z "${MAX_PARALLEL}" ]]; then
  MAX_PARALLEL="${#PRED_LEN_ARRAY[@]}"
fi

if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "Error: --max-parallel must be a positive integer." >&2
  exit 1
fi

run_one_pred_len() {
  local pred_len="$1"
  local case_name="${2:-}"
  local log_stem="pred_len_${pred_len}"
  if [[ -n "${case_name}" ]]; then
    log_stem="${case_name}_${log_stem}"
  fi
  local log_path="${LOG_DIR}/${log_stem}.log"

  (
    cd "${PROJECT_ROOT}"

    export PRED_LENS="${pred_len}"
    if [[ -n "${case_name}" ]]; then
      export RUN_NONE="0"
      export RUN_SOFT_BETA_1="0"
      export RUN_SOFT_BETA_2="0"
      export RUN_HARD="0"

      case "${case_name}" in
        none)
          export RUN_NONE="1"
          ;;
        soft1|soft_beta_1|soft_bias_beta_1)
          export RUN_SOFT_BETA_1="1"
          ;;
        soft2|soft_beta_2|soft_bias_beta_2)
          export RUN_SOFT_BETA_2="1"
          ;;
        hard|hard_matched)
          export RUN_HARD="1"
          ;;
        *)
          echo "Error: unsupported case in parallel.case_matrix: ${case_name}" >&2
          exit 1
          ;;
      esac
    fi

    if [[ -n "${DEVICE_OVERRIDE}" ]]; then
      export DEVICE="${DEVICE_OVERRIDE}"
    fi

    if [[ -n "${NUM_WORKERS_OVERRIDE}" ]]; then
      export NUM_WORKERS="${NUM_WORKERS_OVERRIDE}"
    fi

    if [[ -n "${COMPUTE_THREADS}" ]]; then
      export OMP_NUM_THREADS="${COMPUTE_THREADS}"
      export MKL_NUM_THREADS="${COMPUTE_THREADS}"
      export OPENBLAS_NUM_THREADS="${COMPUTE_THREADS}"
      export NUMEXPR_NUM_THREADS="${COMPUTE_THREADS}"
      export TORCH_NUM_THREADS="${COMPUTE_THREADS}"
    fi

    echo "[$(date '+%F %T')] Start ${EXPERIMENT_LABEL}, pred_len=${pred_len}"
    echo "Experiment script: ${EXPERIMENT_SCRIPT}"
    echo "PRED_LENS=${PRED_LENS}"
    [[ -n "${case_name}" ]] && echo "CASE=${case_name}"
    [[ -n "${DEVICE_OVERRIDE}" ]] && echo "DEVICE=${DEVICE_OVERRIDE}"
    [[ -n "${NUM_WORKERS_OVERRIDE}" ]] && echo "NUM_WORKERS=${NUM_WORKERS_OVERRIDE}"
    [[ -n "${COMPUTE_THREADS}" ]] && echo "COMPUTE_THREADS=${COMPUTE_THREADS}"

    bash "${EXPERIMENT_SCRIPT}"

    echo "[$(date '+%F %T')] Finished ${EXPERIMENT_LABEL}, pred_len=${pred_len}"
  ) >"${log_path}" 2>&1
}

wait_for_batch() {
  local failed=0
  local pid

  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  return "${failed}"
}

echo "======================================================================"
echo "Running pred_len experiments in parallel"
echo "Experiment   -> ${EXPERIMENT_LABEL}"
echo "Script       -> ${EXPERIMENT_SCRIPT}"
echo "Pred lens    -> ${PRED_LEN_ARRAY[*]}"
[[ -n "${CASE_MATRIX}" ]] && echo "Case matrix  -> ${CASE_MATRIX}"
echo "Max parallel -> ${MAX_PARALLEL}"
echo "Log dir      -> ${LOG_DIR}"
[[ -n "${DEVICE_OVERRIDE}" ]] && echo "Device       -> ${DEVICE_OVERRIDE}"
[[ -n "${NUM_WORKERS_OVERRIDE}" ]] && echo "Num workers  -> ${NUM_WORKERS_OVERRIDE}"
[[ -n "${COMPUTE_THREADS}" ]] && echo "Threads/run  -> ${COMPUTE_THREADS}"
echo "======================================================================"

batch_pids=()
failed=0
read -r -a CASE_ARRAY <<< "${CASE_MATRIX//,/ }"

start_run() {
  local pred_len="$1"
  local case_name="${2:-}"
  local log_stem="pred_len_${pred_len}"
  if [[ -n "${case_name}" ]]; then
    log_stem="${case_name}_${log_stem}"
  fi

  run_one_pred_len "${pred_len}" "${case_name}" &
  pid="$!"
  batch_pids+=("${pid}")
  echo "Started pred_len=${pred_len}${case_name:+, case=${case_name}}, pid=${pid}, log=${LOG_DIR}/${log_stem}.log"

  if [[ "${#batch_pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
    if ! wait_for_batch "${batch_pids[@]}"; then
      failed=1
    fi
    batch_pids=()
  fi
}

for pred_len in "${PRED_LEN_ARRAY[@]}"; do
  if [[ "${#CASE_ARRAY[@]}" -gt 0 ]]; then
    for case_name in "${CASE_ARRAY[@]}"; do
      start_run "${pred_len}" "${case_name}"
    done
  else
    start_run "${pred_len}"
  fi
done

if [[ "${#batch_pids[@]}" -gt 0 ]]; then
  if ! wait_for_batch "${batch_pids[@]}"; then
    failed=1
  fi
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "One or more runs failed. Check logs in ${LOG_DIR}." >&2
  exit 1
fi

echo "All parallel pred_len runs finished."
