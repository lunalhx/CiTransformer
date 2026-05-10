#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/project_config.sh"
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

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}
DATA_DIR="${DATA_DIR:-$(project_config_get paths.data_dir)}"
CAUSAL_GRAPH_DIR="${CAUSAL_GRAPH_DIR:-$(project_config_get paths.results.causal_graphs_global_pcmci_11vars_train)}"
CAUSAL_TRAIN_PATH="${CAUSAL_TRAIN_PATH:-${DATA_DIR}/splits/train.csv}"
REBUILD_CAUSAL_GRAPH="${REBUILD_CAUSAL_GRAPH:-0}"
TRUST_EXISTING_CAUSAL_GRAPH="${TRUST_EXISTING_CAUSAL_GRAPH:-0}"
CAUSAL_SAMPLE_SCOPE="${CAUSAL_SAMPLE_SCOPE:-full_train}"
CAUSAL_TAU_MIN="${CAUSAL_TAU_MIN:-1}"
CAUSAL_TAU_MAX="${CAUSAL_TAU_MAX:-12}"
CAUSAL_PC_ALPHA="${CAUSAL_PC_ALPHA:-0.05}"
CAUSAL_ALPHA_LEVEL="${CAUSAL_ALPHA_LEVEL:-0.05}"
CAUSAL_FDR_METHOD="${CAUSAL_FDR_METHOD:-fdr_bh}"
CAUSAL_FREQ_MINUTES="${CAUSAL_FREQ_MINUTES:-5}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

CAUSAL_GRAPH_PATH="$(project_path "${CAUSAL_GRAPH_DIR}")"
CAUSAL_TRAIN_PATH_ABS="$(project_path "${CAUSAL_TRAIN_PATH}")"

causal_graph_config_is_current() {
  local graph_dir="$1"
  local expected_train_path="$2"
  local config_path="${graph_dir}/global_pcmci_config.json"
  local adjacency_path="${graph_dir}/global_causal_adjacency.csv"

  "${PYTHON_BIN}" - \
    "${config_path}" \
    "${adjacency_path}" \
    "${expected_train_path}" \
    "${CAUSAL_SAMPLE_SCOPE}" \
    "${CAUSAL_TAU_MIN}" \
    "${CAUSAL_TAU_MAX}" \
    "${CAUSAL_PC_ALPHA}" \
    "${CAUSAL_ALPHA_LEVEL}" \
    "${CAUSAL_FDR_METHOD}" \
    "${CAUSAL_FREQ_MINUTES}" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
adjacency_path = Path(sys.argv[2])
expected_train_path = os.path.abspath(os.path.expanduser(sys.argv[3]))
expected_sample_scope = sys.argv[4]
expected_tau_min = int(sys.argv[5])
expected_tau_max = int(sys.argv[6])
expected_pc_alpha = float(sys.argv[7])
expected_alpha_level = float(sys.argv[8])
expected_fdr_method = sys.argv[9]
expected_freq_minutes = int(sys.argv[10])

expected_variables = [
    "Active_Pow",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Weather_T",
    "Weather_R",
    "solar_elevation",
    "sin_time_of_day",
    "cos_time_of_day",
    "sin_day_of_year",
    "cos_day_of_year",
    "day_night_label",
]

def fail(reason: str) -> None:
    print(f"Causal graph rebuild required: {reason}", file=sys.stderr)
    raise SystemExit(1)

if not config_path.exists():
    fail(f"missing config {config_path}")
if not adjacency_path.exists():
    fail(f"missing adjacency {adjacency_path}")

with config_path.open("r", encoding="utf-8") as fp:
    config = json.load(fp)

actual_train_path = os.path.abspath(os.path.expanduser(str(config.get("train_path", ""))))
if actual_train_path != expected_train_path:
    fail(f"train_path mismatch: {actual_train_path} != {expected_train_path}")
if config.get("variables") != expected_variables:
    fail("variable order mismatch")
if config.get("sample_scope") != expected_sample_scope:
    fail(f"sample_scope mismatch: {config.get('sample_scope')} != {expected_sample_scope}")
if int(config.get("tau_min", -1)) != expected_tau_min:
    fail(f"tau_min mismatch: {config.get('tau_min')} != {expected_tau_min}")
if int(config.get("tau_max", -1)) != expected_tau_max:
    fail(f"tau_max mismatch: {config.get('tau_max')} != {expected_tau_max}")
if abs(float(config.get("pc_alpha", -1.0)) - expected_pc_alpha) > 1e-12:
    fail(f"pc_alpha mismatch: {config.get('pc_alpha')} != {expected_pc_alpha}")
if abs(float(config.get("alpha_level", -1.0)) - expected_alpha_level) > 1e-12:
    fail(f"alpha_level mismatch: {config.get('alpha_level')} != {expected_alpha_level}")
if config.get("fdr_method") != expected_fdr_method:
    fail(f"fdr_method mismatch: {config.get('fdr_method')} != {expected_fdr_method}")
if int(config.get("freq_minutes", -1)) != expected_freq_minutes:
    fail(f"freq_minutes mismatch: {config.get('freq_minutes')} != {expected_freq_minutes}")

print(f"Reusing current causal graph: {adjacency_path}")
PY
}

if [[ "${TRUST_EXISTING_CAUSAL_GRAPH}" == "1" ]]; then
  if [[ ! -f "${CAUSAL_GRAPH_PATH}/global_pcmci_config.json" || ! -f "${CAUSAL_GRAPH_PATH}/global_causal_adjacency.csv" ]]; then
    echo "Error: TRUST_EXISTING_CAUSAL_GRAPH=1 but causal graph files are incomplete under ${CAUSAL_GRAPH_PATH}" >&2
    exit 1
  fi
  echo "Trusting existing causal graph without path/config rebuild check: ${CAUSAL_GRAPH_PATH}"
elif [[ "${REBUILD_CAUSAL_GRAPH}" == "1" ]] || ! causal_graph_config_is_current "${CAUSAL_GRAPH_PATH}" "${CAUSAL_TRAIN_PATH_ABS}"; then
  "${PYTHON_BIN}" -u "${CAUSAL_SCRIPT}" \
    --train_path "${CAUSAL_TRAIN_PATH}" \
    --sample_scope "${CAUSAL_SAMPLE_SCOPE}" \
    --tau_min "${CAUSAL_TAU_MIN}" \
    --tau_max "${CAUSAL_TAU_MAX}" \
    --pc_alpha "${CAUSAL_PC_ALPHA}" \
    --alpha_level "${CAUSAL_ALPHA_LEVEL}" \
    --fdr_method "${CAUSAL_FDR_METHOD}" \
    --freq_minutes "${CAUSAL_FREQ_MINUTES}" \
    --output_dir "${CAUSAL_GRAPH_DIR}"
fi

PYTHON_BIN="${PYTHON_BIN}" \
MODE="${MODE:-train}" \
DATA_DIR="${DATA_DIR}" \
SEQ_LEN="${SEQ_LEN:-96}" \
PRED_LENS="${PRED_LENS:-1 12 24 48}" \
BATCH_SIZE="${BATCH_SIZE:-256}" \
D_MODEL="${D_MODEL:-128}" \
N_HEADS="${N_HEADS:-4}" \
E_LAYERS="${E_LAYERS:-2}" \
D_FF="${D_FF:-256}" \
FACTOR="${FACTOR:-5}" \
DROPOUT="${DROPOUT:-0.1}" \
ACTIVATION="${ACTIVATION:-gelu}" \
LEARNING_RATE="${LEARNING_RATE:-5e-4}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}" \
GRAD_CLIP="${GRAD_CLIP:-1.0}" \
EPOCHS="${EPOCHS:-80}" \
PATIENCE="${PATIENCE:-15}" \
MIN_DELTA="${MIN_DELTA:-1e-5}" \
NUM_WORKERS="${NUM_WORKERS:-$(project_config_get runtime.num_workers)}" \
SEED="${SEED:-42}" \
DEVICE="${DEVICE:-$(project_config_get runtime.device)}" \
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-$(project_config_get paths.results.itransformer_global_pcmci_11vars)}" \
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$(project_config_get paths.checkpoints.itransformer_global_pcmci_11vars)}" \
bash "${RUN_SCRIPT}" --causal_graph_dir "${CAUSAL_GRAPH_DIR}"
