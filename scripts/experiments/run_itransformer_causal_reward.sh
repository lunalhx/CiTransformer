#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"

RUN_SCRIPT="${PROJECT_ROOT}/scripts/train/run_iTransformer.py"
SUMMARY_SCRIPT="${PROJECT_ROOT}/scripts/reports/summarize_itransformer_causal_reward.py"

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}

if ! "${PYTHON_BIN}" -c "import torch, pandas, matplotlib" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} is missing required packages (torch/pandas/matplotlib)." >&2
  exit 1
fi

setup_matplotlib_cache

DATA_DIR="${DATA_DIR:-$(project_config_get paths.data_dir)}"
RESULTS_ROOT="${RESULTS_ROOT:-$(project_config_get paths.results_root)}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$(project_config_get paths.checkpoints_root)}"

REGIME_MODEL_PATH="${REGIME_MODEL_PATH:-${RESULTS_ROOT}/regimes/gmm_hmm_daytime_k7/gmm_hmm_regime_model.pkl}"
REGIME_GRAPH_ROOT="${REGIME_GRAPH_ROOT:-${RESULTS_ROOT}/causal_graphs/regime_target_pcmci_k7}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${RESULTS_ROOT}/itransformer_transition_weighted_causal_reward}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${CHECKPOINTS_ROOT}/itransformer_transition_weighted_causal_reward}"

PRED_LENS="${PRED_LENS:-12 24 48}"
GAMMAS="${GAMMAS:-0.1 0.25 0.5 1.0 2.0}"
SEQ_LEN="${SEQ_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-256}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"
E_LAYERS="${E_LAYERS:-2}"
D_FF="${D_FF:-256}"
FACTOR="${FACTOR:-5}"
DROPOUT="${DROPOUT:-0.1}"
ACTIVATION="${ACTIVATION:-gelu}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
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
REPORT_SPLIT="${REPORT_SPLIT:-validation}"
TUNING_ONLY="${TUNING_ONLY:-1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
TIME_COL="${TIME_COL:-}"
SAMPLING_FREQ_MINUTES="${SAMPLING_FREQ_MINUTES:-}"

read -r -a PRED_LEN_ARRAY <<< "${PRED_LENS//,/ }"
read -r -a GAMMA_ARRAY <<< "${GAMMAS//,/ }"

cd "${PROJECT_ROOT}"

gamma_dir_name() {
  local gamma="$1"
  "${PYTHON_BIN}" - "${gamma}" <<'PY'
import sys

gamma = float(sys.argv[1])
name = f"gamma_{gamma:.1f}" if gamma.is_integer() else f"gamma_{gamma:g}"
print(name.replace(".", "p"))
PY
}

run_case() {
  local pred_len="$1"
  local gamma="$2"
  local gamma_dir
  gamma_dir="$(gamma_dir_name "${gamma}")"
  local results_dir="${RESULTS_BASE_DIR}/${gamma_dir}/pred_len_${pred_len}"
  local checkpoint_path="${CHECKPOINT_BASE_DIR}/${gamma_dir}/pred_len_${pred_len}/best_model.pth"

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
    --report_split "${REPORT_SPLIT}"
    --causal_mask_mode causal_reward
    --causal_gamma "${gamma}"
    --causal_reward_strength max_abs_mci
    --causal_strength_normalization per_target_max
    --regime_graph_root "${REGIME_GRAPH_ROOT}"
    --regime_model_path "${REGIME_MODEL_PATH}"
    --regime_mask_strategy transition_weighted
  )

  if [[ "${TUNING_ONLY}" == "1" ]]; then
    cmd+=(--tuning_only)
  fi
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

  echo "======================================================================"
  echo "Running iTransformer causal_reward pred_len=${pred_len} gamma=${gamma}"
  echo "Results    -> $(project_path "${results_dir}")"
  echo "Checkpoint -> $(project_path "${checkpoint_path}")"
  echo "======================================================================"
  "${cmd[@]}"
}

for pred_len in "${PRED_LEN_ARRAY[@]}"; do
  for gamma in "${GAMMA_ARRAY[@]}"; do
    run_case "${pred_len}" "${gamma}"
  done
done

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
  --results_root "${RESULTS_ROOT}" \
  --causal_reward_root "${RESULTS_BASE_DIR}" \
  --pred_lens "${PRED_LEN_ARRAY[@]}" \
  --gammas "${GAMMA_ARRAY[@]}"
