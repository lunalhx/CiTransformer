#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/project_config.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment.sh TASK [args...]

Tasks:
  lstm                         Run LSTM pred_len batch experiments
  persistence                  Run persistence baseline pred_len batch experiments
  itransformer                 Run vanilla/masked iTransformer pred_len batch experiments
  itransformer-tuning          Run standard iTransformer tuning
  itransformer-tuned-test      Export/evaluate tuned iTransformer best configs
  global-pcmci-mask            Build or reuse the global PCMCI causal mask only
  global-pcmci-itransformer    Build/reuse global PCMCI mask, then run masked iTransformer
  mask-calibration             Run causal mask calibration cases
  regime-discovery             Run daytime regime discovery
  regime-pcmci                 Run target-regime-conditioned PCMCI
  parallel-pred-lens           Run configured pred_len jobs in parallel

Examples:
  bash scripts/run_experiment.sh --help
  bash scripts/run_experiment.sh lstm
  bash scripts/run_experiment.sh global-pcmci-mask
  bash scripts/run_experiment.sh regime-pcmci --help
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 1
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
esac

TASK="$1"
shift

PYTHON_BIN="$(resolve_python_bin)" || {
  echo "Error: no usable Python interpreter found. Set PYTHON_BIN manually." >&2
  exit 1
}
export PYTHON_BIN

case "${TASK}" in
  lstm)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_lstm_experiments.sh" "$@"
    ;;
  persistence)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_persistence_experiments.sh" "$@"
    ;;
  itransformer)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_itransformer_experiments.sh" "$@"
    ;;
  itransformer-tuning)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_itransformer_tuning_standard.sh" "$@"
    ;;
  itransformer-tuned-test)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_itransformer_tuned_test.sh" "$@"
    ;;
  global-pcmci-mask)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_global_pcmci_mask.sh" "$@"
    ;;
  global-pcmci-itransformer)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_global_pcmci_itransformer_11vars.sh" "$@"
    ;;
  mask-calibration)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_itransformer_mask_calibration.sh" "$@"
    ;;
  regime-discovery)
    exec "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/causal/run_gmm_hmm_daytime_regimes.py" "$@"
    ;;
  regime-pcmci)
    exec "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/causal/run_regime_target_pcmci.py" "$@"
    ;;
  parallel-pred-lens)
    exec bash "${PROJECT_ROOT}/scripts/experiments/run_parallel_pred_lens.sh" "$@"
    ;;
  *)
    echo "Error: unknown task: ${TASK}" >&2
    usage >&2
    exit 1
    ;;
esac
