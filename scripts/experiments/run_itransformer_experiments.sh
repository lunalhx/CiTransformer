#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/train/run_iTransformer.py"

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
MODE="${MODE:-train}"
REPORT_SPLIT="${REPORT_SPLIT:-test}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-1 12 24 48}"
RUN_PRED_LEN1_REF="${RUN_PRED_LEN1_REF:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"
E_LAYERS="${E_LAYERS:-2}"
D_FF="${D_FF:-256}"
FACTOR="${FACTOR:-5}"
DROPOUT="${DROPOUT:-0.1}"
ACTIVATION="${ACTIVATION:-gelu}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"
MIN_DELTA="${MIN_DELTA:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-$(project_config_get runtime.num_workers)}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"
PROGRESS_MININTERVAL="${PROGRESS_MININTERVAL:-15}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-$(project_config_get runtime.device)}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-$(project_config_get paths.results.itransformer)}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$(project_config_get paths.checkpoints.itransformer)}"
TUNING_PLAN="${TUNING_PLAN:-standard}"
TUNING_RESULTS_ROOT="${TUNING_RESULTS_ROOT:-$(project_config_get paths.results.tuning_itransformer)/${TUNING_PLAN}}"
TUNING_SUMMARY_ROOT="${TUNING_SUMMARY_ROOT:-${TUNING_RESULTS_ROOT}/summary}"
TIME_COL="${TIME_COL:-}"
SAMPLING_FREQ_MINUTES="${SAMPLING_FREQ_MINUTES:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
DISABLE_NORM="${DISABLE_NORM:-0}"
OUTPUT_ATTENTION="${OUTPUT_ATTENTION:-0}"
CAUSAL_MASK_MODE="${CAUSAL_MASK_MODE:-hard}"
CAUSAL_MASK_BETA="${CAUSAL_MASK_BETA:-1.0}"
REGIME_GRAPH_ROOT="${REGIME_GRAPH_ROOT:-}"
REGIME_LABEL_DIR="${REGIME_LABEL_DIR:-}"
REGIME_COL="${REGIME_COL:-regime}"
REGIME_MASK_SELECTION="${REGIME_MASK_SELECTION:-input_end}"

if [[ "${MODE}" != "train" && "${MODE}" != "export_tuned_best" ]]; then
  echo "Error: MODE must be one of: train, export_tuned_best" >&2
  exit 1
fi

if [[ "${MODE}" == "export_tuned_best" && "${RESULTS_BASE_DIR}" == "$(project_config_get paths.results.itransformer)" ]]; then
  RESULTS_BASE_DIR="$(project_config_get paths.results.itransformer_tuned)"
fi

read -r -a PRED_LEN_ARRAY <<< "${PRED_LENS//,/ }"

if [[ "${#PRED_LEN_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: PRED_LENS is empty." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

resolve_tuned_best_checkpoint() {
  local pred_len="$1"
  local summary_root
  summary_root="$(project_path "${TUNING_SUMMARY_ROOT}")"

  "${PYTHON_BIN}" - "${summary_root}" "${pred_len}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_root = Path(sys.argv[1]).resolve()
pred_len = int(sys.argv[2])

if pred_len == 1:
    ranking_path = summary_root / "with_pred_len_1" / "ranking_by_pred_len.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"Cannot find {ranking_path}. Re-run tuning with RUN_PRED_LEN1_REF=1 before exporting pred_len=1."
        )
    with ranking_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    candidate_rows = [
        row for row in rows
        if int(float(row["pred_len"])) == 1 and row.get("tuning_stage") == "ref1"
    ]
    if not candidate_rows:
        candidate_rows = [row for row in rows if int(float(row["pred_len"])) == 1]
    if not candidate_rows:
        raise RuntimeError("No pred_len=1 reference run found in with_pred_len_1/ranking_by_pred_len.csv")
    metrics_path = Path(candidate_rows[0]["metrics_path"])
else:
    best_config_path = summary_root / "final_validation" / "best_shared_config.json"
    all_runs_path = summary_root / "final_validation" / "all_runs.csv"
    if not best_config_path.exists():
        raise FileNotFoundError(f"Cannot find {best_config_path}. Run tuning first.")
    if not all_runs_path.exists():
        raise FileNotFoundError(f"Cannot find {all_runs_path}. Run tuning first.")

    with best_config_path.open("r", encoding="utf-8") as fp:
        best_config = json.load(fp)
    signature = best_config.get("shared_signature")
    if not signature:
        raise RuntimeError(f"{best_config_path} does not contain shared_signature")

    with all_runs_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    candidate_rows = [
        row for row in rows
        if int(float(row["pred_len"])) == pred_len and row.get("shared_signature") == signature
    ]
    if not candidate_rows:
        raise RuntimeError(
            f"No run for pred_len={pred_len} matches the best shared signature from {best_config_path}"
        )
    metrics_path = Path(candidate_rows[0]["metrics_path"])

with metrics_path.open("r", encoding="utf-8") as fp:
    payload = json.load(fp)
checkpoint_path = payload.get("checkpoint_path")
if not checkpoint_path:
    raise RuntimeError(f"{metrics_path} does not contain checkpoint_path")
print(checkpoint_path)
PY
}

run_train_case() {
  local pred_len="$1"
  local results_dir="${RESULTS_BASE_DIR}/pred_len_${pred_len}"
  local checkpoint_path="${CHECKPOINT_BASE_DIR}/pred_len_${pred_len}/best_model.pth"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${RUN_SCRIPT}"
    --data_dir "${DATA_DIR}"
    --seq_len "${SEQ_LEN}"
    --pred_len "${pred_len}"
    --batch_size "${BATCH_SIZE}"
    --d_model "${D_MODEL}"
    --n_heads "${N_HEADS}"
    --e_layers "${E_LAYERS}"
    --d_ff "${D_FF}"
    --factor "${FACTOR}"
    --activation "${ACTIVATION}"
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

  if [[ "${DISABLE_NORM}" == "1" ]]; then
    cmd+=(--disable_norm)
  fi

  if [[ "${OUTPUT_ATTENTION}" == "1" ]]; then
    cmd+=(--output_attention)
  fi

  cmd+=(--causal_mask_mode "${CAUSAL_MASK_MODE}")
  cmd+=(--causal_mask_beta "${CAUSAL_MASK_BETA}")
  if [[ -n "${REGIME_GRAPH_ROOT}" ]]; then
    cmd+=(--regime_graph_root "${REGIME_GRAPH_ROOT}")
    cmd+=(--regime_col "${REGIME_COL}")
    cmd+=(--regime_mask_selection "${REGIME_MASK_SELECTION}")
  fi
  if [[ -n "${REGIME_LABEL_DIR}" ]]; then
    cmd+=(--regime_label_dir "${REGIME_LABEL_DIR}")
  fi

  if [[ "$#" -gt 1 ]]; then
    cmd+=("${@:2}")
  fi

  echo "======================================================================"
  echo "Running iTransformer baseline with pred_len=${pred_len}"
  echo "Project    -> ${PROJECT_ROOT}"
  echo "Python     -> ${PYTHON_BIN}"
  echo "Data dir   -> $(project_path "${DATA_DIR}")"
  echo "Results    -> $(project_path "${results_dir}")"
  echo "Checkpoint -> $(project_path "${checkpoint_path}")"
  echo "======================================================================"

  "${cmd[@]}"
}

run_export_case() {
  local pred_len="$1"

  if [[ "${pred_len}" == "1" && "${RUN_PRED_LEN1_REF}" != "1" ]]; then
    echo "Skipping pred_len=1 export because RUN_PRED_LEN1_REF=${RUN_PRED_LEN1_REF}"
    return 0
  fi

  local checkpoint_path
  checkpoint_path="$(resolve_tuned_best_checkpoint "${pred_len}")"
  local results_dir="${RESULTS_BASE_DIR}/pred_len_${pred_len}"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${RUN_SCRIPT}"
    --eval_checkpoint_path "${checkpoint_path}"
    --data_dir "${DATA_DIR}"
    --results_dir "${results_dir}"
    --report_split "${REPORT_SPLIT}"
    --device "${DEVICE}"
    --num_workers "${NUM_WORKERS}"
    --causal_mask_mode "${CAUSAL_MASK_MODE}"
    --causal_mask_beta "${CAUSAL_MASK_BETA}"
    --progress_mininterval "${PROGRESS_MININTERVAL}"
    --experiment_name "tuned_sharedbest_pred_len_${pred_len}"
  )
  if [[ -n "${REGIME_GRAPH_ROOT}" ]]; then
    cmd+=(--regime_graph_root "${REGIME_GRAPH_ROOT}")
    cmd+=(--regime_col "${REGIME_COL}")
    cmd+=(--regime_mask_selection "${REGIME_MASK_SELECTION}")
  fi
  if [[ -n "${REGIME_LABEL_DIR}" ]]; then
    cmd+=(--regime_label_dir "${REGIME_LABEL_DIR}")
  fi

  if [[ -n "${TIME_COL}" ]]; then
    cmd+=(--time_col "${TIME_COL}")
  fi

  if [[ -n "${SAMPLING_FREQ_MINUTES}" ]]; then
    cmd+=(--sampling_freq_minutes "${SAMPLING_FREQ_MINUTES}")
  fi

  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    cmd+=(--max_eval_batches "${MAX_EVAL_BATCHES}")
  fi

  echo "======================================================================"
  echo "Exporting tuned iTransformer predictions with pred_len=${pred_len}"
  echo "Project    -> ${PROJECT_ROOT}"
  echo "Python     -> ${PYTHON_BIN}"
  echo "Report     -> ${REPORT_SPLIT}"
  echo "Results    -> $(project_path "${results_dir}")"
  echo "Checkpoint -> $(project_path "${checkpoint_path}")"
  echo "======================================================================"

  "${cmd[@]}"
}

for pred_len in "${PRED_LEN_ARRAY[@]}"; do
  if [[ "${MODE}" == "train" ]]; then
    run_train_case "${pred_len}" "$@"
  else
    run_export_case "${pred_len}"
  fi
done

if [[ "${MODE}" == "train" ]]; then
  echo "All iTransformer experiments finished."
else
  echo "All tuned iTransformer exports finished."
fi
