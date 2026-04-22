#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_iTransformer.py"
SUMMARY_SCRIPT="${PROJECT_ROOT}/scripts/summarize_itransformer_tuning.py"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Error: cannot find ${RUN_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${SUMMARY_SCRIPT}" ]]; then
  echo "Error: cannot find ${SUMMARY_SCRIPT}" >&2
  exit 1
fi

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi

  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    echo "${PROJECT_ROOT}/.venv/bin/python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  return 1
}

slugify_value() {
  local value="$1"
  value="${value//./p}"
  value="${value// /}"
  echo "${value}"
}

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}

if ! "${PYTHON_BIN}" -c "import torch, pandas, matplotlib" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} is missing required packages (torch/pandas/matplotlib)." >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/citransformer-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

PLAN="${PLAN:-minimal}"
if [[ "${PLAN}" != "minimal" && "${PLAN}" != "standard" ]]; then
  echo "Error: PLAN must be one of: minimal, standard" >&2
  exit 1
fi

DATA_DIR="${DATA_DIR:-data/processed}"
SEQ_LEN="${SEQ_LEN:-96}"
MAIN_PRED_LENS="${MAIN_PRED_LENS:-12 24}"
RUN_PRED_LEN1_REF="${RUN_PRED_LEN1_REF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

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
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
MIN_DELTA="${MIN_DELTA:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"
PROGRESS_MININTERVAL="${PROGRESS_MININTERVAL:-15}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
TIME_COL="${TIME_COL:-}"
SAMPLING_FREQ_MINUTES="${SAMPLING_FREQ_MINUTES:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"

TUNING_RESULTS_ROOT="${TUNING_RESULTS_ROOT:-results/tuning/itransformer/${PLAN}}"
TUNING_CHECKPOINT_ROOT="${TUNING_CHECKPOINT_ROOT:-checkpoints/tuning/itransformer/${PLAN}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${TUNING_RESULTS_ROOT}/summary}"

read -r -a MAIN_PRED_LEN_ARRAY <<< "${MAIN_PRED_LENS//,/ }"
if [[ "${#MAIN_PRED_LEN_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: MAIN_PRED_LENS is empty." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

run_experiment() {
  local stage="$1"
  local label="$2"
  local pred_len="$3"
  local seq_len="$4"
  local learning_rate="$5"
  local batch_size="$6"
  local d_model="$7"
  local d_ff="$8"
  local e_layers="$9"
  local dropout="${10}"
  local weight_decay="${11}"
  local activation="${12}"
  local n_heads="${13}"
  local factor="${14}"

  local run_name
  run_name=\
"${stage}_${label}_pl${pred_len}_lr$(slugify_value "${learning_rate}")_bs${batch_size}_dm${d_model}_ff${d_ff}_el${e_layers}_do$(slugify_value "${dropout}")_wd$(slugify_value "${weight_decay}")_act${activation}_nh${n_heads}_fac${factor}"

  local results_dir="${TUNING_RESULTS_ROOT}/pred_len_${pred_len}/${run_name}"
  local checkpoint_path="${TUNING_CHECKPOINT_ROOT}/pred_len_${pred_len}/${run_name}/best_model.pth"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${PROJECT_ROOT}/${results_dir}/metrics.json" ]]; then
    echo "Skipping existing run -> ${run_name}"
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -u "${RUN_SCRIPT}"
    --data_dir "${DATA_DIR}"
    --seq_len "${seq_len}"
    --pred_len "${pred_len}"
    --batch_size "${batch_size}"
    --d_model "${d_model}"
    --n_heads "${n_heads}"
    --e_layers "${e_layers}"
    --d_ff "${d_ff}"
    --factor "${factor}"
    --activation "${activation}"
    --dropout "${dropout}"
    --learning_rate "${learning_rate}"
    --weight_decay "${weight_decay}"
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
    --report_split validation
    --tuning_only
    --experiment_name "${run_name}"
    --tuning_stage "${stage}"
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

  echo "======================================================================"
  echo "Running ${stage} / ${label} / pred_len=${pred_len}"
  echo "Results    -> ${PROJECT_ROOT}/${results_dir}"
  echo "Checkpoint -> ${PROJECT_ROOT}/${checkpoint_path}"
  echo "======================================================================"

  "${cmd[@]}"
}

run_matrix() {
  local stage="$1"
  local matrix="$2"
  local line

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue

    local label seq_len learning_rate batch_size d_model d_ff e_layers dropout weight_decay activation n_heads factor
    IFS=$'\t' read -r label seq_len learning_rate batch_size d_model d_ff e_layers dropout weight_decay activation n_heads factor <<< "${line}"

    local pred_len
    for pred_len in "${MAIN_PRED_LEN_ARRAY[@]}"; do
      run_experiment \
        "${stage}" \
        "${label}" \
        "${pred_len}" \
        "${seq_len}" \
        "${learning_rate}" \
        "${batch_size}" \
        "${d_model}" \
        "${d_ff}" \
        "${e_layers}" \
        "${dropout}" \
        "${weight_decay}" \
        "${activation}" \
        "${n_heads}" \
        "${factor}"
    done
  done <<< "${matrix}"
}

stage1_matrix() {
  if [[ "${PLAN}" == "minimal" ]]; then
    cat <<EOF
baseline	${SEQ_LEN}	${LEARNING_RATE}	${BATCH_SIZE}	${D_MODEL}	${D_FF}	${E_LAYERS}	${DROPOUT}	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
lr5e4	${SEQ_LEN}	5e-4	${BATCH_SIZE}	${D_MODEL}	${D_FF}	${E_LAYERS}	${DROPOUT}	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
batch128	${SEQ_LEN}	${LEARNING_RATE}	128	${D_MODEL}	${D_FF}	${E_LAYERS}	${DROPOUT}	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
drop0p0	${SEQ_LEN}	${LEARNING_RATE}	${BATCH_SIZE}	${D_MODEL}	${D_FF}	${E_LAYERS}	0.0	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
EOF
    return 0
  fi

  cat <<EOF
baseline	${SEQ_LEN}	${LEARNING_RATE}	${BATCH_SIZE}	${D_MODEL}	${D_FF}	${E_LAYERS}	${DROPOUT}	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
lr5e4	${SEQ_LEN}	5e-4	${BATCH_SIZE}	${D_MODEL}	${D_FF}	${E_LAYERS}	${DROPOUT}	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
batch128	${SEQ_LEN}	${LEARNING_RATE}	128	${D_MODEL}	${D_FF}	${E_LAYERS}	${DROPOUT}	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
drop0p0	${SEQ_LEN}	${LEARNING_RATE}	${BATCH_SIZE}	${D_MODEL}	${D_FF}	${E_LAYERS}	0.0	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
drop0p2	${SEQ_LEN}	${LEARNING_RATE}	${BATCH_SIZE}	${D_MODEL}	${D_FF}	${E_LAYERS}	0.2	${WEIGHT_DECAY}	${ACTIVATION}	${N_HEADS}	${FACTOR}
EOF
}

stage2_matrix() {
  local best_seq_len="$1"
  local best_learning_rate="$2"
  local best_batch_size="$3"
  local best_d_model="$4"
  local best_d_ff="$5"
  local best_e_layers="$6"
  local best_dropout="$7"
  local best_weight_decay="$8"
  local best_activation="$9"
  local best_n_heads="${10}"
  local best_factor="${11}"

  local wider_d_model=192
  local wider_d_ff=384
  local deeper_e_layers=3
  local ff4x_d_ff=$(( best_d_model * 4 ))

  if [[ "${PLAN}" == "minimal" ]]; then
    cat <<EOF
wider192	${best_seq_len}	${best_learning_rate}	${best_batch_size}	${wider_d_model}	${wider_d_ff}	${best_e_layers}	${best_dropout}	${best_weight_decay}	${best_activation}	${best_n_heads}	${best_factor}
deeper3	${best_seq_len}	${best_learning_rate}	${best_batch_size}	${best_d_model}	${best_d_ff}	${deeper_e_layers}	${best_dropout}	${best_weight_decay}	${best_activation}	${best_n_heads}	${best_factor}
EOF
    return 0
  fi

  cat <<EOF
wider192	${best_seq_len}	${best_learning_rate}	${best_batch_size}	${wider_d_model}	${wider_d_ff}	${best_e_layers}	${best_dropout}	${best_weight_decay}	${best_activation}	${best_n_heads}	${best_factor}
ff4x	${best_seq_len}	${best_learning_rate}	${best_batch_size}	${best_d_model}	${ff4x_d_ff}	${best_e_layers}	${best_dropout}	${best_weight_decay}	${best_activation}	${best_n_heads}	${best_factor}
deeper3	${best_seq_len}	${best_learning_rate}	${best_batch_size}	${best_d_model}	${best_d_ff}	${deeper_e_layers}	${best_dropout}	${best_weight_decay}	${best_activation}	${best_n_heads}	${best_factor}
EOF
}

summarize_runs() {
  local output_dir="$1"
  shift
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --root_dir "${TUNING_RESULTS_ROOT}" \
    --output_dir "${output_dir}" \
    --pred_lens "${MAIN_PRED_LEN_ARRAY[@]}" \
    "$@"
}

best_config_tsv() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --root_dir "${TUNING_RESULTS_ROOT}" \
    --output_dir "${SUMMARY_ROOT}/tmp" \
    --pred_lens "${MAIN_PRED_LEN_ARRAY[@]}" \
    "$@" \
    --print_best_tsv
}

echo "======================================================================"
echo "iTransformer tuning plan -> ${PLAN}"
echo "Validation-only tuning targets pred_len -> ${MAIN_PRED_LENS}"
echo "Primary objective -> daytime RMSE / MAE on validation for pred_len=12 and 24"
echo "Results root -> ${PROJECT_ROOT}/${TUNING_RESULTS_ROOT}"
echo "======================================================================"

run_matrix "s1" "$(stage1_matrix)"
summarize_runs "${PROJECT_ROOT}/${SUMMARY_ROOT}/stage1" --stage_filter s1

best_stage1_tsv="$(best_config_tsv --stage_filter s1)"
IFS=$'\t' read -r \
  best_seq_len \
  best_learning_rate \
  best_batch_size \
  best_d_model \
  best_d_ff \
  best_e_layers \
  best_dropout \
  best_weight_decay \
  best_activation \
  best_n_heads \
  best_factor \
  best_disable_norm \
  best_output_attention \
  best_seed <<< "${best_stage1_tsv}"

if [[ "${best_disable_norm}" == "True" || "${best_output_attention}" == "True" ]]; then
  echo "Error: tuning helper expects disable_norm/output_attention to stay fixed at False." >&2
  exit 1
fi

run_matrix \
  "s2" \
  "$(stage2_matrix \
    "${best_seq_len}" \
    "${best_learning_rate}" \
    "${best_batch_size}" \
    "${best_d_model}" \
    "${best_d_ff}" \
    "${best_e_layers}" \
    "${best_dropout}" \
    "${best_weight_decay}" \
    "${best_activation}" \
    "${best_n_heads}" \
    "${best_factor}")"

summarize_runs "${PROJECT_ROOT}/${SUMMARY_ROOT}/final_validation"

if [[ "${RUN_PRED_LEN1_REF}" == "1" ]]; then
  best_overall_tsv="$(best_config_tsv)"
  IFS=$'\t' read -r \
    best_seq_len \
    best_learning_rate \
    best_batch_size \
    best_d_model \
    best_d_ff \
    best_e_layers \
    best_dropout \
    best_weight_decay \
    best_activation \
    best_n_heads \
    best_factor \
    best_disable_norm \
    best_output_attention \
    best_seed <<< "${best_overall_tsv}"

  run_experiment \
    "ref1" \
    "sharedbest" \
    "1" \
    "${best_seq_len}" \
    "${best_learning_rate}" \
    "${best_batch_size}" \
    "${best_d_model}" \
    "${best_d_ff}" \
    "${best_e_layers}" \
    "${best_dropout}" \
    "${best_weight_decay}" \
    "${best_activation}" \
    "${best_n_heads}" \
    "${best_factor}"

  summarize_runs "${PROJECT_ROOT}/${SUMMARY_ROOT}/with_pred_len_1"
fi

echo
echo "Validation-only tuning finished."
echo "Inspect summary files under ${PROJECT_ROOT}/${SUMMARY_ROOT}"
echo "- stage1 ranking: ${PROJECT_ROOT}/${SUMMARY_ROOT}/stage1/ranking_shared_configs.csv"
echo "- final ranking: ${PROJECT_ROOT}/${SUMMARY_ROOT}/final_validation/ranking_shared_configs.csv"
if [[ "${RUN_PRED_LEN1_REF}" == "1" ]]; then
  echo "- final ranking with pred_len=1 reference: ${PROJECT_ROOT}/${SUMMARY_ROOT}/with_pred_len_1/ranking_by_pred_len.csv"
fi
echo
echo "No test metrics were used in this tuning script."
